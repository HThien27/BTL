#khai báo thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

# Đọc dữ liệu
df = pd.read_csv("student_data.csv")
df.head()#hiện 5 dòng đầu

df.info()#thông tin trong dữ liệu
df = df.set_index('ID')#hợp nhất ID trog dataset với số tt
df.describe()#mô tả vè dữ liệu

# số lượng giới tính trog data
gender_counts = df['gender'].value_counts()
# Print the counts
print(gender_counts)
# Create a bar plot
plt.figure(figsize=(8, 6))
sns.countplot(x='gender', data=df)
plt.title('Distribution of Gender')
plt.xlabel('Gender (0: Non-binary, 1: Male, 2: Female)')
plt.ylabel('Count')
plt.show()

# Mã hóa cột giới tính
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
# Mã hóa cột năm học
def transform_year(year):
  if year == 'Freshman':
    return 1
  elif year == 'Sophomore':
    return 2
  elif year == 'Junior':
    return 3
  elif year == 'Senior':
    return 4
  else:
    return None

df['year_in_school'] = df['year_in_school'].apply(transform_year)
# Mã hóa cột phương thức thanh toán
df['preferred_payment_method'] = le.fit_transform(df['preferred_payment_method'])
df.head()

# Tính toán phần trăm giá trị thiếu cho mỗi cột
missing_percentage = (df.isnull().sum() / len(df))
# In ra màn hình
print(missing_percentage)

#chia học phí trung bình hàng tháng
df['tuition'] = df['tuition'] / 12
df.head()

#Tổng thu của sinh viên hàng tháng
df['total_income'] = df['monthly_income'] + df['financial_aid']

#tổng chi phí đã chi tiêu hàng tháng
expense_cols = ['tuition', 'housing', 'food', 'transportation', 'books_supplies',
                'entertainment', 'personal_care', 'technology', 'health_wellness',
                'miscellaneous']
df['total_expenses'] = df[expense_cols].sum(axis=1)

# tổng chi tiêu hàng tháng
df['net_income'] = df['total_income'] - df['total_expenses']

# thêm cột vào dataset
df[['total_income', 'total_expenses', 'net_income']]

#trực quan hóa thu và chi
# Count students with positive and negative net income
positive_net_income_count = len(df[df['net_income'] >= 0])
negative_net_income_count = len(df[df['net_income'] < 0])
print(f"Số lượng sinh viên có khoản thu dương: {positive_net_income_count}")
print(f"Số lượng sinh viên có khoản thu âm: {negative_net_income_count}")
# Calculate the percentage of students with positive and negative net income
total_students = len(df)
positive_percentage = (positive_net_income_count / total_students) * 100
negative_percentage = (negative_net_income_count / total_students) * 100

# tạo sơ đồ trực qua hóa
labels = ['Thu dương', 'Thu âm']
sizes = [positive_percentage, negative_percentage]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)  # Explode the 'Thu dương' slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Tỷ lệ sinh viên có khoản thu chi dương và âm')
plt.show()

#trực quan hóa về phương thức thanh toán
payment_method_counts = df['preferred_payment_method'].value_counts()
# Create a bar plot (swo đồ 1)
plt.figure(figsize=(10, 6))
sns.barplot(x=payment_method_counts.index, y=payment_method_counts.values)
plt.title('Distribution of Preferred Payment Methods')
plt.xlabel('Payment Method (Encoded)')
plt.ylabel('Number of Students')
# Annotate each bar with its count
for i, v in enumerate(payment_method_counts.values):
    plt.text(i, v + 1, str(v), ha='center', va='bottom')

plt.show()

print("( 0: cash, 1:Credit/Debit Card , 2: Mobile Payment App)")
original_payment_methods = ['Cash', 'Credit Card', 'Debit Card'] 

payment_method_proportions = payment_method_counts / payment_method_counts.sum()

# Create the pie chart(sơ đồ 2)
plt.figure(figsize=(8, 8))
plt.pie(payment_method_proportions, labels=original_payment_methods, autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Preferred Payment Methods')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#tương quan giữ tỏng thu và chi 
correlation = df['total_income'].corr(df['total_expenses'])

# Print the correlation
print(f"Tỉ lệ tương quan giữa tổng thu và chi: {correlation}")

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(8, 6))
sns.scatterplot(x='total_income', y='total_expenses', data=df)
plt.title('Mối quan hệ giữa Tổng thu và Tổng chi')
plt.xlabel('Tổng thu')
plt.ylabel('Tổng chi')
plt.show()

#train nháp mô hình máy học hồi quy về tương quan thu và chi
# Thực hiện phân tích hồi quy giữa Tổng thu và Tổng chi
# Chuẩn bị dữ liệu cho mô hình hồi quy
X = df[['total_income']]  # Biến độc lập: tổng thu nhập
y = df['total_expenses']  # Biến phụ thuộc: tổng chi phí

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán tổng chi phí trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}") # Sai số bình phương trung bình, càng nhỏ càng tốt
print(f"R-squared (R2): {r2}") # Hệ số xác định, càng gần 1 càng tốt, thể hiện phần trăm phương sai của biến phụ thuộc được giải thích bởi mô hình

# Hiển thị các thông số của mô hình
print(f"Hệ số hồi quy (slope): {model.coef_[0]}") # Hệ số góc của đường thẳng hồi quy, cho biết sự thay đổi của tổng chi khi tổng thu thay đổi 1 đơn vị
print(f"Intercept: {model.intercept_}") # Giao điểm của đường thẳng hồi quy với trục tung, tổng chi khi tổng thu bằng 0


# Trực quan hóa kết quả
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Dữ liệu thực tế') # Vẽ điểm dữ liệu thực tế
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Đường hồi quy') # Vẽ đường hồi quy
plt.title('Hồi quy tuyến tính giữa Tổng thu và Tổng chi')
plt.xlabel('Tổng thu nhập')
plt.ylabel('Tổng chi phí')
plt.legend() # Hiển thị chú thích
plt.show()


#Giải thích:
# Mô hình hồi quy tuyến tính được sử dụng để dự đoán tổng chi phí dựa trên tổng thu nhập.
# Hệ số hồi quy thể hiện sự thay đổi trung bình của tổng chi phí khi tổng thu nhập tăng thêm 1 đơn vị.  Một hệ số dương cho thấy mối quan hệ trực tiếp (tổng thu tăng thì tổng chi cũng tăng), và ngược lại.
# Intercept là giá trị dự đoán của tổng chi phí khi tổng thu nhập bằng 0.  Trong ngữ cảnh này, có thể không có ý nghĩa thực tế.
# MSE là sai số bình phương trung bình, đo lường sự khác biệt giữa giá trị dự đoán và giá trị thực tế. Giá trị MSE càng thấp, mô hình càng chính xác.
# R-squared (R2) là hệ số xác định, cho biết phần trăm phương sai của biến phụ thuộc (tổng chi) được giải thích bởi mô hình (tổng thu). Giá trị R2 càng cao, mô hình càng tốt, càng giải thích được nhiều biến thiên của tổng chi phí.
# Đồ thị thể hiện đường hồi quy và các điểm dữ liệu thực tế, giúp trực quan hóa mối quan hệ giữa tổng thu và tổng chi.


#nháp train mô hình hồi quy tuyens tính so sánh hiệu quả dự đoán chi tiêu
# Chuẩn bị dữ liệu cho mô hình Rừng ngẫu nhiên
X = df[['total_income', 'monthly_income', 'financial_aid', 'tuition', 'housing', 'food', 'transportation', 'books_supplies', 'entertainment', 'personal_care', 'technology', 'health_wellness', 'miscellaneous']]  # Các biến độc lập
y = df['total_expenses'] # Biến phụ thuộc

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình Rừng ngẫu nhiên
rf_model = RandomForestRegressor(n_estimators=100, random_state=42) # n_estimators: số cây trong rừng
rf_model.fit(X_train, y_train)

# Dự đoán tổng chi phí trên tập kiểm tra
rf_y_pred = rf_model.predict(X_test)

# Đánh giá mô hình Rừng ngẫu nhiên
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print(f"Random Forest - Mean Squared Error (MSE): {rf_mse}")
print(f"Random Forest - R-squared (R2): {rf_r2}")

# So sánh hiệu quả giữa hai mô hình (Hồi quy tuyến tính và Rừng ngẫu nhiên)
print("\nSo sánh hiệu quả:")
print(f"Hồi quy tuyến tính - MSE: {mse}, R2: {r2}")
print(f"Rừng ngẫu nhiên - MSE: {rf_mse}, R2: {rf_r2}")

#nháp trainmoo hình tìm các nhóm sinh viên có hành vi chi tiêu tương tự.
# Chọn các thuộc tính liên quan đến hành vi chi tiêu
spending_features = ['tuition', 'housing', 'food', 'transportation', 'books_supplies',
                      'entertainment', 'personal_care', 'technology', 'health_wellness',
                      'miscellaneous']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_spending_data = scaler.fit_transform(df[spending_features])

# Xác định số lượng cụm (k) tối ưu bằng phương pháp Elbow
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_spending_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Chọn số lượng cụm dựa trên phương pháp Elbow (ví dụ: k=3)
optimal_k = 3  # Thay đổi giá trị này nếu cần thiết
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['spending_cluster'] = kmeans.fit_predict(scaled_spending_data)

# Hiển thị kết quả
print(df.groupby('spending_cluster')[spending_features].mean())

#nháp train mô hình dự đoán chi tiêu hàng thnags sinh viên