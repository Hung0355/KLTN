import os

def delete_txt_files(root_folder):
    # Lặp qua tất cả các thư mục và tập tin trong thư mục gốc
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            # Kiểm tra xem file có phần mở rộng là .txt không
            if file.endswith('.txt'):
                file_path = os.path.join(dirpath, file)
                try:
                    # Xóa file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

# Đường dẫn tới thư mục gốc
root_folder = 'D:\Anaconda\Cheating Scenario Dataset in Online Exam'

# Gọi hàm để xóa các file .txt
delete_txt_files(root_folder)
