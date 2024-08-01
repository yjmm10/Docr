FROM python:3.10

LABEL name="Telos"
LABEL version="0.0.2"
LABEL description="🛠️  Component design with module-based functionality, allowing for on-demand feature acquisition,🚀 easy to expand, and flexible to use, just like playing with building blocks! | 🛠️ 组件化设计，让功能模块化，实现按需获取，🚀 易于扩展，使用起来灵活方便，就像搭积木一样简单！"

WORKDIR /app

ADD . ./

# CMD ["python"]