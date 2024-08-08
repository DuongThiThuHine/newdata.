import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def line_map(datax, datay, title, xlbl, ylbl):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(datax, datay)
    ax.set_title(title)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    st.pyplot(fig)

def bar_chart(datax,datay,title,xlbl,ylbl):
    plt.figure(figsize=(20, 10))
    plt.bar(datax, datay)
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.show()    
def pie_chart(datax,data_lbl,title):
    plt.figure(figsize=(10, 6))
    plt.pie(datax, labels=data_lbl, autopct='%1.1f%%', startangle=140)   
    plt.title(title)    
    plt.show()  
def histogram_chart(datax,bins,title,xlbl,ylbl):
    plt.figure(figsize=(10, 6))
    plt.hist(datax, bins=bins, edgecolor='black')  # Bins có thể được điều chỉnh theo yêu cầu
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.grid(True)
    plt.show()


st.write("""
# Xin chào
## Xin chào
### Xin chào
""")   #>streamlit run new.py
data=pd.read_csv('train.csv')
# Upload CSV file
# st.header('Upload CSV')
# data=pd.DataFrame()
# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
data.info()
st.write(data)

# lay Order Date va Sales, Sum giá trị Sales theo Order Date
#chuyen doi cot Order Date thành Datetime
data['Order Date']=pd.to_datetime(data['Order Date'],format='%d/%m/%Y')# định dạng ngày tháng theo : dd/MM/yyyy
#data.head()
#data.info()
df1=data.groupby('Order Date')['Sales'].sum().reset_index()
df1.info()

st.line_chart(df1,x='Order Date',y='Sales') # theo thang , theo quy theo nam
line_map(df1['Order Date'],df1['Sales'],'dat cai gì đó','Ngày','Nhiệt Độ') # theo thang , theo quy theo nam

data['Year Month'] = data['Order Date'].apply(lambda x: x.strftime('%Y-%m'))  # theo nam
df2 = pd.DataFrame(data.groupby('Year Month')['Sales'].sum().reset_index())
df2.head(5)
line_map(df2['Year Month'],df2['Sales'],'dat cai gì đó','X labale','Y Lable')

data['Year Quarter'] = data['Order Date'].dt.to_period('Q')
df3= pd.DataFrame(data.groupby('Year Quarter')['Sales'].sum().reset_index())
df3.head(5)
line_map(df3['Year Quarter'].astype(str),df3['Sales'],'dat cai gì đó','X labale','Y Lable')

st.bar_chart(df3, x='Year Quarter',y=['Sales'])

data['Year'] = data['Order Date'].dt.year
df4 = data.groupby('Year')['Sales'].sum().reset_index()
df4
kpi_data = pd.DataFrame({
    'Year': [2015, 2016,2017,2018],
    'KPI': [450000, 500000,550000,620000]
})
# Kết hợp doanh số theo năm và KPI
df4_end = pd.merge(df4, kpi_data, on='Year')
df4_end
st.bar_chart(df4_end, x='Year',y=['KPI','Sales'])
# Tạo cột 'Year' từ cột 'Order Date'
data['Year'] = data['Order Date'].dt.year
# Tính tổng doanh số bán hàng theo năm
df4 = data.groupby('Year')['Sales'].sum().reset_index()
# Dữ liệu KPI
kpi_data = pd.DataFrame({
    'Year': [2015, 2016, 2017, 2018],
    'KPI': [450000, 500000, 550000, 620000]
})
# Kết hợp doanh số theo năm và KPI
df4_end = pd.merge(df4, kpi_data, on='Year')
# Tạo biểu đồ thanh đa màu sắc với Plotly
fig = go.Figure(data=[
    go.Bar(name='Sales', x=df4_end['Year'], y=df4_end['Sales'], marker_color='blue', text=df4_end['Sales'], textposition='auto'),
    go.Bar(name='KPI', x=df4_end['Year'], y=df4_end['KPI'], marker_color='orange', text=df4_end['KPI'], textposition='auto')
])
# Thay đổi bố cục để các thanh nằm cạnh nhau
fig.update_layout(barmode='group', title='Sales vs KPI by Year')
# Hiển thị biểu đồ trong Streamlit
st.title("Biểu Đồ Thanh Đa Màu Sắc với Giá Trị Trên Cột")
st.plotly_chart(fig)

df7=data.groupby('City')['Sales'].sum().reset_index()
df7=df7.sort_values(by='Sales',ascending=False)
df7=df7[df7['Sales']<500000]
df7
# Vẽ biểu đồ Hitogram
bins=[10000,20000,30000,40000,50000]  ## bins=5 bin='auto'
histogram_chart(df7['Sales'],bins,'Sales Distribution','Sales','Frequency')
# Tính tổng doanh số bán hàng theo thành phố
df7 = data.groupby('City')['Sales'].sum().reset_index()
df7 = df7.sort_values(by='Sales', ascending=False)
df7 = df7[df7['Sales'] < 50000]  # Lọc các giá trị Sales < 50000

# Đặt bins
bins = [10000, 20000, 30000, 40000, 50000]

# Tạo biểu đồ histogram với Plotly
fig = fig.histogram(df7, x='Sales', nbins=len(bins), title='Sales Distribution', labels={'Sales':'Sales', 'count':'Frequency'})
fig.update_traces(xbins=dict(start=min(bins), end=max(bins), size=(max(bins) - min(bins)) / len(bins)))

# Hiển thị biểu đồ trong Streamlit
st.title("Sales Distribution Histogram")
st.plotly_chart(fig)

