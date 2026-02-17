from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import yaml
from PIL import Image
import uuid
import albumentations as A
from albumentations.pytorch import ToTensorV2

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODELS_FOLDER = 'models'

# Model paths
YOLO_MODEL_PATH = os.path.join(MODELS_FOLDER, 'best.pt')
UNET_MODEL_PATH = os.path.join(MODELS_FOLDER, 'UNET_ultra_new.pt')
UNETPP_MODEL_PATH = os.path.join(MODELS_FOLDER, 'UNETPlusPlus_ultra.pt')
CONFIG_PATH = os.path.join(MODELS_FOLDER, 'dataset_config.yaml')

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

def addPadding(srcShapeTensor, tensor_whose_shape_isTobechanged):
    if srcShapeTensor.shape != tensor_whose_shape_isTobechanged.shape:
        target = torch.zeros(srcShapeTensor.shape)
        target[:, :, :tensor_whose_shape_isTobechanged.shape[2],
               :tensor_whose_shape_isTobechanged.shape[3]] = tensor_whose_shape_isTobechanged
        return target.to(DEVICE)
    return tensor_whose_shape_isTobechanged.to(DEVICE)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)
        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv_4 = double_conv(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, image):
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        x = self.up_trans_1(x9)
        x = addPadding(x7, x)
        x = self.up_conv_1(torch.cat([x7, x], 1))

        x = self.up_trans_2(x)
        x = addPadding(x5, x)
        x = self.up_conv_2(torch.cat([x5, x], 1))

        x = self.up_trans_3(x)
        x = addPadding(x3, x)
        x = self.up_conv_3(torch.cat([x3, x], 1))

        x = self.up_trans_4(x)
        x = addPadding(x1, x)
        x = self.up_conv_4(torch.cat([x1, x], 1))

        return self.out(x)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False):
        super().__init__()
        
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output

class ModelManager:
    def __init__(self):
        self.yolo_model = None
        self.unet_model = None
        self.unetpp_model = None
        self.class_names = {}
        self.load_models()
    
    def load_models(self):
        print(" Loading models...")
        
        # Load YOLO model
        try:
            if os.path.exists(YOLO_MODEL_PATH):
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                print(" YOLO model loaded")
                
                # Load class names
                if os.path.exists(CONFIG_PATH):
                    with open(CONFIG_PATH, 'r') as f:
                        config = yaml.safe_load(f)
                    self.class_names = config.get('names', {0: 'Brachial plexus nerve', 1: 'Vagus nerve'})
                else:
                    self.class_names = {0: 'Brachial plexus nerve', 1: 'Vagus nerve'}
            else:
                print(" YOLO model not found")
        except Exception as e:
            print(f" Error loading YOLO model: {e}")
        
        # Load UNet model
        try:
            if os.path.exists(UNET_MODEL_PATH):
                self.unet_model = UNet().to(DEVICE)
                self.unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE))
                self.unet_model.eval()
                print(" UNet model loaded")
            else:
                print(" UNet model not found")
        except Exception as e:
            print(f" Error loading UNet model: {e}")
        
        # Load UNet++ model
        try:
            if os.path.exists(UNETPP_MODEL_PATH):
                self.unetpp_model = UNetPlusPlus().to(DEVICE)
                self.unetpp_model.load_state_dict(torch.load(UNETPP_MODEL_PATH, map_location=DEVICE))
                self.unetpp_model.eval()
                print(" UNet++ model loaded")
            else:
                print(" UNet++ model not found")
        except Exception as e:
            print(f" Error loading UNet++ model: {e}")

model_manager = ModelManager()

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def load_and_convert_image(image_path):
    try:
        file_ext = get_file_extension(image_path)
        
        if file_ext == '.tif':
            pil_img = Image.open(image_path)
            if pil_img.mode == 'L':
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'P':
                pil_img = pil_img.convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                pil_img = Image.open(image_path)
                if pil_img.mode == 'L':
                    pil_img = pil_img.convert('RGB')
                elif pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Ensure 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_rgb_version(image_path, rgb_path):
    try:
        img = load_and_convert_image(image_path)
        if img is None:
            return False
        cv2.imwrite(rgb_path, img)
        return True
    except Exception as e:
        print(f"Error creating RGB version: {e}")
        return False

def save_image_with_detections(original_image_path, detections, output_path):
    try:
        img = load_and_convert_image(original_image_path)
        if img is None:
            return False
        
        colors = {0: (0, 0, 255), 1: (0, 255, 0)}
        
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            class_name = detection['class']
            confidence = detection['confidence']
            
            color = colors.get(class_id, (255, 255, 255))
            
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(img, 
                         (bbox[0], bbox[1] - text_height - 15),
                         (bbox[0] + text_width + 10, bbox[1]),
                         color, -1)
            
            cv2.putText(img, label, 
                       (bbox[0] + 5, bbox[1] - 5),
                       font, font_scale, (255, 255, 255), thickness)
        
        file_ext = get_file_extension(output_path)
        
        if file_ext == '.tif':
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(img_rgb)
            pil_result.save(output_path, format='TIFF')
        else:
            cv2.imwrite(output_path, img)
        
        return True
    except Exception as e:
        print(f"Error saving image with detections: {e}")
        return False

def save_segmentation_result(original_image_path, mask, output_path, model_name):
    try:
        img = load_and_convert_image(original_image_path)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        mask_resized = cv2.resize(mask, (w, h))
        
        overlay = np.zeros_like(img)
        overlay[:, :, 1] = mask_resized * 255  
        
        result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        cv2.putText(result, f"{model_name} Segmentation", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        file_ext = get_file_extension(output_path)
        
        if file_ext == '.tif':
            img_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(img_rgb)
            pil_result.save(output_path, format='TIFF')
        else:
            cv2.imwrite(output_path, result)
        
        return True
    except Exception as e:
        print(f"Error saving segmentation result: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yolo-detection')
def yolo_detection():
    return render_template('yolo_detection.html')

@app.route('/segmentation')
def segmentation():
    return render_template('segmentation.html')

@app.route('/api/yolo-predict', methods=['POST'])
def yolo_predict():
    if not model_manager.yolo_model:
        return jsonify({'error': 'YOLO model not loaded'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        original_filename = file.filename
        file_ext = get_file_extension(original_filename)
        unique_id = str(uuid.uuid4())[:8]
        filename = f"upload_{unique_id}{file_ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        file.save(filepath)
        
        rgb_filename = f"rgb_{unique_id}.jpg"
        rgb_filepath = os.path.join(UPLOAD_FOLDER, rgb_filename)
        
        if not create_rgb_version(filepath, rgb_filepath):
            return jsonify({'error': 'Failed to process image format'})
        
        confidence = float(request.form.get('confidence', 0.25))
        results = model_manager.yolo_model.predict(source=rgb_filepath, conf=confidence, save=False, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'class_id': cls,
                    'class': model_manager.class_names.get(cls, f'Class_{cls}'),
                    'confidence': round(conf, 3),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
        
        result_filename = f"yolo_result_{unique_id}{file_ext}"
        result_filepath = os.path.join(RESULTS_FOLDER, result_filename)
        
        if detections:
            save_success = save_image_with_detections(filepath, detections, result_filepath)
        else:
            if file_ext == '.tif':
                img = load_and_convert_image(filepath)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_result = Image.fromarray(img_rgb)
                    pil_result.save(result_filepath, format='TIFF')
                    save_success = True
                else:
                    save_success = False
            else:
                import shutil
                shutil.copy2(filepath, result_filepath)
                save_success = True
        
        if os.path.exists(rgb_filepath):
            os.remove(rgb_filepath)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'uploaded_image': filename,
            'result_image': result_filename if save_success else None,
            'original_filename': original_filename
        })
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        if 'rgb_filepath' in locals() and os.path.exists(rgb_filepath):
            os.remove(rgb_filepath)
        return jsonify({'error': str(e)})

@app.route('/api/segmentation-predict', methods=['POST'])
def segmentation_predict():
    if not model_manager.unet_model and not model_manager.unetpp_model:
        return jsonify({'error': 'Segmentation models not loaded'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        original_filename = file.filename
        file_ext = get_file_extension(original_filename)
        unique_id = str(uuid.uuid4())[:8]
        filename = f"upload_{unique_id}{file_ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        file.save(filepath)
        
        img = load_and_convert_image(filepath)
        if img is None:
            return jsonify({'error': 'Failed to load image'})
        
        transform = A.Compose([
            A.Resize(128, 128),
            A.Normalize(mean=[0.0]*3, std=[1.0]*3, max_pixel_value=255.0),
            ToTensorV2()
        ])
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = transform(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
        
        results = {}
        
        # UNet prediction
        if model_manager.unet_model:
            with torch.no_grad():
                unet_output = model_manager.unet_model(transformed)
                unet_output = torch.sigmoid(unet_output).squeeze().cpu().numpy()
                unet_mask = (unet_output > 0.5).astype(np.uint8)
                
                unet_result_filename = f"unet_result_{unique_id}{file_ext}"
                unet_result_filepath = os.path.join(RESULTS_FOLDER, unet_result_filename)
                unet_success = save_segmentation_result(filepath, unet_mask, unet_result_filepath, "UNet")
                
                results['unet'] = {
                    'result_image': unet_result_filename if unet_success else None,
                    'mask_area': float(np.sum(unet_mask)),
                    'mask_percentage': float(np.mean(unet_mask) * 100)
                }
        
        # UNet++ prediction
        if model_manager.unetpp_model:
            with torch.no_grad():
                unetpp_output = model_manager.unetpp_model(transformed)
                if isinstance(unetpp_output, list):
                    unetpp_output = unetpp_output[-1]
                unetpp_output = torch.sigmoid(unetpp_output).squeeze().cpu().numpy()
                unetpp_mask = (unetpp_output > 0.5).astype(np.uint8)
                
                # Save UNet++ result
                unetpp_result_filename = f"unetpp_result_{unique_id}{file_ext}"
                unetpp_result_filepath = os.path.join(RESULTS_FOLDER, unetpp_result_filename)
                unetpp_success = save_segmentation_result(filepath, unetpp_mask, unetpp_result_filepath, "UNet++")
                
                results['unetpp'] = {
                    'result_image': unetpp_result_filename if unetpp_success else None,
                    'mask_area': float(np.sum(unetpp_mask)),
                    'mask_percentage': float(np.mean(unetpp_mask) * 100)
                }
        
        return jsonify({
            'success': True,
            'uploaded_image': filename,
            'original_filename': original_filename,
            'results': results
        })
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    try:
        data = request.json
        uploaded_file = data.get('uploaded_file')
        
        if uploaded_file:
            filepath = os.path.join(UPLOAD_FOLDER, uploaded_file)
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print(" Starting Advanced Nerve Detection & Segmentation App")
    print(f" Device: {DEVICE}")
    print(f" YOLO Model: {' Loaded' if model_manager.yolo_model else ' Not found'}")
    print(f" UNet Model: {' Loaded' if model_manager.unet_model else ' Not found'}")
    print(f" UNet++ Model: {' Loaded' if model_manager.unetpp_model else ' Not found'}")
    app.run(debug=True, host='0.0.0.0', port=5000)