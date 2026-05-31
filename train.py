import os
from ultralytics import YOLO

# 1. تحديد المسار المطلق للمجلد الحالي الذي يحتوي على المستودع
current_dir = os.path.abspath(os.path.dirname(__file__))

# 2. بناء المسار المطلق لملف data.yaml
data_yaml_path = os.path.join(current_dir, 'data.yaml')

# 3. تحميل النموذج
model = YOLO('yolo11n.pt')

# 4. بدء التدريب مع إرسال المسار الكامل وتحديد مجلد العمل الحالي
if __name__ == '__main__':
    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,
        device='cpu',
        project=os.path.join(current_dir, 'runs') # حفظ النتائج داخل المستودع أيضًا
    )
