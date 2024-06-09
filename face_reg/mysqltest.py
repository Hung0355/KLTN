import mysql.connector
import wx

def show_message():
    app = wx.App(False)
    wx.MessageBox('matching :))))!', 'Thông báo', wx.OK | wx.ICON_INFORMATION)


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

selectedExam = "A100"
selectedSeat = "C3"
# Thực hiện truy vấn
query = f"SELECT uid FROM `{selectedExam}` WHERE seat = %s"
cursor.execute(query,(selectedSeat,))

# Lấy dữ liệu
results = cursor.fetchall()
# if results:
#     uid_val =  results[0]

for row in results:
    print(row[0])
    if row[0] == '12348765':
        show_message()

# Đóng kết nối
cursor.close()
cnx.close()
