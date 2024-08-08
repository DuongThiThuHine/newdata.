import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.title("Data Analysis about Vietnamcovid for Web App")
# Đọc tệp CSV với mã hóa 'latin1'
df = pd.read_csv('vietnamcovid.csv', encoding='latin1')
st.write('Read data from file csv')
st.write(df)
st.write('Data info of columns')
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

df[df.duplicated()] # xem dữ liệu trùng

st.write(df.describe())

df.isnull().sum()

df.isnull()

df.notnull()

df.fillna(0)

st.write(df.columns)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.info()

def load_data(file_path, file_type='csv'):
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'excel':
        return pd.read_excel(file_path)
    elif file_type == 'sql':
        # Let's say you have established a database connection
        return pd.read_sql(file_path, con=load_data)
    else:
        raise ValueError("Unsupported file type!")
    
def inspect_data(df):
    st.write("Shape of DataFrame:", df.shape)
    st.write("Data Types:\n", df.dtypes)
    st.write("First few rows:\n", df.head())
    st.write("Missing values:\n", df.isnull().sum())

def clean_data(df):
    # Xử lý giá trị thiếu
    df = df.dropna(how='all')
    df['Product'] = df['Product'].fillna('Unknown')
    df['Price'] = df['Price'].fillna(df['Price'].mean())

    # Chuyển đổi cột 'Date' sang kiểu datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])

    # Loại bỏ các hàng có giá trị số âm
    df = df[df['Quantity'] > 0]

    # Chuẩn hóa các giá trị trong cột 'Product'
    df['Product'] = df['Product'].str.strip().str.upper()

    return df

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    return outliers

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    return df_cleaned

from sklearn.preprocessing import MinMaxScaler

def normalize_data(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
def encode_categorical(df, columns):
    df = pd.get_dummies(df, columns=columns)
    return df
def feature_engineering(df):
    df['Total Value'] = df['Quantity'] * df['Price']
    return df
from sklearn.model_selection import train_test_split

def split_data(df, target, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df, columns):
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()
import pandas as pd

# Function to analyze gender ratio
def gender_ratio(df):
    gender_counts = df['Gender'].value_counts()
    total_cases = len(df)
    male_ratio = gender_counts.get('Male', 0) / total_cases * 100
    female_ratio = gender_counts.get('Female', 0) / total_cases * 100
    st.write(f"Male: {male_ratio:.2f}%")
    st.write(f"Female: {female_ratio:.2f}%")
    return male_ratio, female_ratio
gender_ratio(df)

# Vẽ biểu đồ sử dụng Matplotlib

# Function to visualize gender ratio using Streamlit
def visualize_gender_ratio(male_ratio, female_ratio):
    labels = ['Male', 'Female']
    sizes = [male_ratio, female_ratio]
    colors = ['lightblue', 'lightpink']
    explode = (0.1, 0)  # explode the 1st slice (Male)
    
    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Gender Ratio of COVID-19 Cases')
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying to prevent overlap

# Main Streamlit app
def main():
    st.title("Gender Ratio Visualization of COVID-19 Cases")

    male_ratio, female_ratio = gender_ratio(df)
    visualize_gender_ratio(male_ratio, female_ratio)

if __name__ == "__main__":
    main()



# Function to analyze locations
def analyze_locations(df):
    location_counts = df['Location'].value_counts()
    st.write(location_counts)
    return location_counts
location_counts = analyze_locations(df)

# Function to visualize locations using Matplotlib for more control
def visualize_locations_bar_chart(location_counts, top_n=10):
    top_locations = location_counts.head(top_n)
    top_locations_df = top_locations.reset_index()  # Convert to DataFrame
    top_locations_df.columns = ['Location', 'Number of Cases']  # Rename columns
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_locations_df['Location'], top_locations_df['Number of Cases'], color='skyblue')
    plt.xlabel('Location')
    plt.ylabel('Number of Cases')
    plt.title('Top Locations by Number of COVID-19 Cases')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying to prevent overlap

# Use the function visualize_locations_bar_chart
location_counts = df['Location'].value_counts()
visualize_locations_bar_chart(location_counts, top_n=10) 

# st.scatter_chart(data_new,x='Order Date',y='Sales') 

# Function to analyze nationalities
def analyze_nationalities(df):
    nationality_counts = df['Nationality'].value_counts()
    return nationality_counts

nationality_counts = analyze_nationalities(df)

# Function to visualize nationalities as a column chart with multiple colors
def visualize_nationalities_bar_chart(nationality_counts, top_n=10):
    top_nationalities = nationality_counts.head(top_n)
    top_nationalities_df = top_nationalities.reset_index()
    top_nationalities_df.columns = ['Nationality', 'Number of Cases']
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_nationalities_df['Nationality'], top_nationalities_df['Number of Cases'], color='skyblue')
    plt.xlabel('Nationality')
    plt.ylabel('Number of Cases')
    plt.title('Top Nationalities by Number of COVID-19 Cases')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying to prevent overlap

# Use the function visualize_nationalities_bar_chart
visualize_nationalities_bar_chart(nationality_counts, top_n=10)

# Function to analyze statuses
def analyze_statuses(df):
    status_counts = df['status'].value_counts()
    st.write(status_counts)
    return status_counts

status_counts = analyze_statuses(df)

# Function to visualize statuses as a horizontal bar chart with multiple colors
def visualize_statuses(status_counts):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Paired(range(len(status_counts)))  # Use a colormap to generate different colors
    status_counts.plot(kind='barh', color=colors)
    plt.title('COVID-19 Case Statuses')
    plt.xlabel('Number of Cases')
    plt.ylabel('Status')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest count on top
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying to prevent overlap

# Use the function visualize_statuses
visualize_statuses(status_counts)

# Function to analyze sources
def analyze_sources(df):
    source_counts = df['Source'].value_counts()
    st.write(source_counts)
    return source_counts
source_counts = analyze_sources(df)

# Function to analyze sources
def analyze_sources(df):
    if 'Source' not in df.columns:
        st.error("The DataFrame does not contain a 'Source' column.")
        return pd.Series(dtype=int)  # Return an empty Series if the column is missing
    return df['Source'].value_counts()

source_counts = analyze_sources(df)

# Function to visualize sources as a horizontal bar chart with multiple colors
def visualize_sources(source_counts, top_n=10):
    if source_counts.empty:
        st.warning("No source data available to visualize.")
        return

    top_sources = source_counts.head(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Paired(range(len(top_sources)))  # Use a colormap to generate different colors
    top_sources.plot(kind='barh', color=colors)
    plt.title(f'Top {top_n} Sources of COVID-19 Infections')
    plt.xlabel('Number of Cases')
    plt.ylabel('Source')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest count on top
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying to prevent overlap

# Use the function visualize_sources
visualize_sources(source_counts, top_n=10)

# Function to analyze related contacts
def analyze_related(df):
    related_counts = df['Related'].value_counts()
    st.write(related_counts)
    return related_counts
related_counts = analyze_related(df)

# Function to analyze related contacts
def analyze_related(df):
    if 'Related' not in df.columns:
        st.error("The DataFrame does not contain a 'Related' column.")
        return pd.Series(dtype=int)  # Return an empty Series if the column is missing
    return df['Related'].value_counts()

related_counts = analyze_related(df)

# Function to visualize related contacts as a column chart
def visualize_related(related_counts, top_n=10):
    if related_counts.empty:
        st.warning("No related contact data available to visualize.")
        return

    top_related = related_counts.head(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Paired(range(len(top_related)))  # Use a colormap to generate different colors
    top_related.plot(kind='bar', color=colors)
    plt.title(f'Top {top_n} Related Contacts of COVID-19 Infections')
    plt.xlabel('Related')
    plt.ylabel('Number of Cases')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying to prevent overlap

# Use the function visualize_related
visualize_related(related_counts, top_n=10)