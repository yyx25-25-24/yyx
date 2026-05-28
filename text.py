# 定义一个供应商类
class Supplier:
    """供应商"""

    def __init__(self, name, contact="", phone="", address=""):
        self.name = name
        self.contact = contact
        self.phone = phone
        self.address = address

    def __str__(self):
        return f"供应商: {self.name} | 联系人: {self.contact} | 电话: {self.phone} | 地址: {self.address}"


if __name__ == "__main__":
    s = Supplier("示例供应商", "张三", "13800138000", "北京市朝阳区")
    print(s)
# 创建一个供应商示例
supplier = Supplier("新供应商", "李四", "13911112222", "上海市浦东新区")
print(supplier)
import pandas as pd
import os

# 如果文件不存在，就创建
if not os.path.exists("suppliers.xlsx"):
    df = pd.DataFrame(columns=["id", "name", "contact", "address"])
    df.to_excel("suppliers.xlsx", index=False)
    print("文件创建成功")

# 添加3个供应商
data = [
    {"id": 1, "name": "合肥电子", "contact": "张经理", "address": "蜀山区"},
    {"id": 2, "name": "南京科技", "contact": "李经理", "address": "南京市"},
    {"id": 3, "name": "上海贸易", "contact": "王经理", "address": "上海市"}
]

df = pd.DataFrame(data)
df.to_excel("suppliers.xlsx", index=False)
print("供应商添加成功")

# 读取并打印
df = pd.read_excel("suppliers.xlsx")
print("\n所有供应商：")
print(df)

# 保存备份成CSV
df.to_csv("suppliers_backup.csv", index=False, encoding="utf-8-sig")
print("备份成功")