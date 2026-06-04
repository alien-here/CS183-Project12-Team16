import cv2
import os
from image_process import process_image, draw_text_boxes
from ocr import recognize_text, get_text_boxes
from utils import save_batch_result, get_all_images
from image_process import process_image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
from matplotlib.widgets import Button



#Author:Chen Zeyue
#Function:run_single
def run_single():
    img_path = "images/清晰/test.jpg"
    import numpy as np
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    processed_img = process_image(img)
    if img is None:
     print(f"错误：无法读取图片 {img_path}")
     exit()
    
    processed = process_image(img)
    text = recognize_text(processed_img)
    boxes = get_text_boxes(processed)
    result_img = draw_text_boxes(img.copy(), boxes)
    
    cv2.imwrite("results/result.jpg", result_img)
    save_batch_result(text, "result.txt")
    
    print("单张识别完成！")
    print("识别结果：")
    print(text)


#Author:Chen Zeyue
#Function:batch_recognition
def batch_recognition():
    print("=== 批量图片识别 ===")
    all_images = get_all_images("images")
    if not all_images:
        print("❌ images 文件夹里没有找到任何图片")
        return
    print(f"📸 共找到 {len(all_images)} 张图片")

    for i, img_path in enumerate(all_images, 1):
        print(f"\n--- 正在处理第 {i}/{len(all_images)} 张：{img_path} ---")
        try:
            import numpy as np
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            processed_img = process_image(img)
            text = recognize_text(processed_img)
            boxes = get_text_boxes(processed_img)
            result_img = draw_text_boxes(img.copy(), boxes)
            cv2.imwrite(f"results/{os.path.basename(img_path)}", result_img)
            save_batch_result(text, img_path)
        except Exception as e:
            print(f"❌ 处理失败：{e}")
    print("\n🎉 所有图片处理完成！结果保存在 results/batch 文件夹里")


#Author:Chen Zeyue
#Function:font_inference
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title('Text Recognition System', fontsize=16, pad=20)
    ax.axis('off')

    
    ax_btn1 = plt.axes([0.2, 0.6, 0.6, 0.18])
    ax_btn2 = plt.axes([0.2, 0.3, 0.6, 0.18])

    btn1 = Button(ax_btn1, 'Single Image Recognition', hovercolor='lightblue')
    btn2 = Button(ax_btn2, 'Batch Image Recognition', hovercolor='lightblue')

    
    btn1.on_clicked(lambda e: (plt.close(), run_single()))
    btn2.on_clicked(lambda e: (plt.close(), batch_recognition()))

    plt.show()