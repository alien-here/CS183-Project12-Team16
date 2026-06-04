import os
#Author:Chen Jiahao
#Function:get_all_images
def get_all_images(folder="images"):
    """获取 images 文件夹（含子文件夹）里所有图片路径"""
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return image_paths
#Author:Chen Jiahao
#Function:save_batch_result
def save_batch_result(text, image_path, output_folder="results/batch"):
    """把单张图片的识别结果保存成独立的txt"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base_name = os.path.basename(image_path).split(".")[0]
    file_path = os.path.join(output_folder, f"{base_name}_result.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ 结果已保存：{file_path}")