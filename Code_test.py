import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu từ file CSV
data = pd.read_csv(r"C:\\Users\\Admin\\Desktop\\UTH\\N2_1\\Python\\BTL\\FileTest.csv")

# In ra các cột có trong DataFrame
print("Các cột trong DataFrame:", data.columns)

# Kiểm tra và loại bỏ khoảng trắng trong tên cột
data.columns = data.columns.str.strip()

# Xác định biến đầu vào (X) và biến mục tiêu (y)
# Sửa lại tên cột nếu cần thiết
X = data[['Income', 'Tuition', 'Living']]  # Sử dụng tên cột đúng
y = data['Spend']

# Chia dữ liệu thành tập train và tập test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình
model = LinearRegression()

# Train mô hình trên tập train
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình bằng Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Dự đoán chi tiêu mới
new_data = [[1400, 350, 550]]
predicted_spend = model.predict(new_data)
print(f"Dự đoán chi tiêu: {predicted_spend[0]}")
