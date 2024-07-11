import cv2
import mysql.connector

# Kết nối đến cơ sở dữ liệu
cnx = mysql.connector.connect(
    host='103.241.43.129',    # Địa chỉ máy chủ MySQL
    user='hung', # Tên đăng nhập
    password='0355', # Mật khẩu
    database='exam', # Tên cơ sở dữ liệu
    ssl_disabled=True,  # Disable SSL
    use_pure=True
)

# Tạo một đối tượng cursor
cursor = cnx.cursor()

# Khởi tạo camera
cap = cv2.VideoCapture(0)
i = 0
nam = 'A'
uid = '20521406'
while True:
    i = i + 1
    # Đọc một khung hình từ camera
    ret, frame = cap.read()

    # Hiển thị khung hình
    cv2.imshow('Camera', frame)

    # Chờ 1 phím bấm
    key = cv2.waitKey(1) & 0xFF

    # Nếu phím 's' được nhấn (save)
    if key == ord('s'):
        # Chuyển đổi hình ảnh thành dữ liệu nhị phân
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = buffer.tobytes()

        # Thực hiện truy vấn chèn hình ảnh vào cơ sở dữ liệu
        query = "INSERT INTO cheating (name,uid,image_data) VALUES (%s,%s,%s)"
        cursor.execute(query, (nam,uid,image_data,))

        # Đồng ý thay đổi
        cnx.commit()

        print("done.")

    # Nếu phím 'q' được nhấn (quit)
    elif key == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

# Đóng kết nối
cursor.close()
cnx.close()
