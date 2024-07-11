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

ROOM = 'C101'
SEAT = '4'
# Thực hiện truy vấn
query = f"SELECT UID, Name FROM ExamRooms WHERE seat = %s AND period = %s AND code = %s"
cursor.execute(query,(SEAT,'1',ROOM))
uid = cursor.fetchone()

# Lấy dữ liệu

# if results:
#     uid_val =  results[0]

print (uid[0])

# Đóng kết nối
cursor.close()
cnx.close()
