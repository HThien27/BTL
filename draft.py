# Khai báo thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# Đọc dữ liệu và hiển thị thông tin cơ bản
df = pd.read_csv("student_data.csv")
print(df.head())  # Hiển thị 5 dòng đầu
df.info()  # Thông tin chi tiết của dữ liệu
df = df.set_index('ID')  # Đặt 'ID' làm chỉ mục
print(df.describe())  # Mô tả thống kê về dữ liệu

# Phân tích phân phối giới tính
gender_counts = df['gender'].value_counts()
print(gender_counts)  # Số lượng mỗi giới tính
plt.figure(figsize=(8, 6))
sns.countplot(x='gender', data=df)
plt.title('Phân phối giới tính')
plt.xlabel('Giới tính (0: Non-binary, 1: Nam, 2: Nữ)')
plt.ylabel('Số lượng')
plt.show()

# Mã hóa các cột dạng chuỗi
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
def transform_year(year):
    return {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4}.get(year, None)
df['year_in_school'] = df['year_in_school'].apply(transform_year)
df['preferred_payment_method'] = le.fit_transform(df['preferred_payment_method'])
print(df.head())

# Tính phần trăm giá trị thiếu
missing_percentage = (df.isnull().sum() / len(df))
print(missing_percentage)

# Chuyển đổi học phí hàng năm sang hàng tháng
df['tuition'] = df['tuition'] / 12
df['total_income'] = df['monthly_income'] + df['financial_aid']
expense_cols = ['tuition', 'housing', 'food', 'transportation', 'books_supplies',
                'entertainment', 'personal_care', 'technology', 'health_wellness', 'miscellaneous']
df['total_expenses'] = df[expense_cols].sum(axis=1)
df['net_income'] = df['total_income'] - df['total_expenses']
print(df[['total_income', 'total_expenses', 'net_income']].head())

# Trực quan hóa thu nhập dương và âm
positive_net_income_count = len(df[df['net_income'] >= 0])
negative_net_income_count = len(df[df['net_income'] < 0])
print(f"Số lượng sinh viên thu nhập dương: {positive_net_income_count}")
print(f"Số lượng sinh viên thu nhập âm: {negative_net_income_count}")

positive_percentage = (positive_net_income_count / len(df)) * 100
negative_percentage = (negative_net_income_count / len(df)) * 100
plt.pie([positive_percentage, negative_percentage], labels=['Thu dương', 'Thu âm'], colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Tỷ lệ thu chi dương và âm')
plt.show()

# Trực quan hóa phương thức thanh toán
payment_method_counts = df['preferred_payment_method'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=payment_method_counts.index, y=payment_method_counts.values)
plt.title('Phân phối phương thức thanh toán')
plt.xlabel('Phương thức thanh toán')
plt.ylabel('Số lượng sinh viên')
plt.show()

# Phân tích tương quan giữa thu nhập và chi tiêu
correlation = df['total_income'].corr(df['total_expenses'])
print(f"Tỉ lệ tương quan giữa tổng thu và chi: {correlation}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_income', y='total_expenses', data=df)
plt.title('Tương quan giữa Tổng thu nhập và Tổng chi tiêu')
plt.xlabel('Tổng thu nhập')
plt.ylabel('Tổng chi tiêu')
plt.show()

# Mô hình hồi quy tuyến tính dự đoán chi tiêu
X = df[['total_income']]
y = df['total_expenses']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Hồi quy tuyến tính - MSE: {mse}, R2: {r2}")
print(f"Hệ số hồi quy: {linear_model.coef_[0]}, Intercept: {linear_model.intercept_}")

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Dữ liệu thực tế')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Đường hồi quy')
plt.title('Hồi quy tuyến tính giữa Tổng thu nhập và Tổng chi tiêu')
plt.xlabel('Tổng thu nhập')
plt.ylabel('Tổng chi tiêu')
plt.legend()
plt.show()

# Mô hình Rừng ngẫu nhiên
X = df[['total_income', 'monthly_income', 'financial_aid'] + expense_cols]
y = df['total_expenses']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
print(f"Rừng ngẫu nhiên - MSE: {rf_mse}, R2: {rf_r2}")

# Phân cụm sinh viên theo hành vi chi tiêu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[expense_cols])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Phương pháp Elbow')
plt.xlabel('Số cụm')
plt.ylabel('WCSS')
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['spending_cluster'] = kmeans.fit_predict(scaled_data)
print(df.groupby('spending_cluster')[expense_cols].mean())
