from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date, timedelta
import os, json, csv, random, string
import pytz
from io import StringIO, BytesIO
import barcode
from barcode.writer import ImageWriter
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///orders.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# パス設定
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# データベースモデル
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False)
    image = db.Column(db.String(200))
    stock_quantity = db.Column(db.Integer, default=0)
    deadline_time = db.Column(db.Time)
    show_date = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.String(20), unique=True, nullable=False)
    customer_name = db.Column(db.String(100), nullable=False)
    phone_number = db.Column(db.String(20))
    grade = db.Column(db.Integer)
    class_num = db.Column(db.Integer)
    number = db.Column(db.Integer)
    total_price = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, ready, completed
    order_date = db.Column(db.Date, default=date.today)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    items = db.relationship('OrderItem', backref='order', lazy=True, cascade='all, delete-orphan')

class OrderItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)

    product = db.relationship('Product', backref='order_items')

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class SalesInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    sale_date = db.Column(db.Date, nullable=False)
    quantity_sold = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    product = db.relationship('Product', backref='sales_records')

class AppConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    require_phone = db.Column(db.Boolean, default=False)
    require_grade = db.Column(db.Boolean, default=False)
    require_class = db.Column(db.Boolean, default=False)
    require_number = db.Column(db.Boolean, default=False)
    max_order_amount = db.Column(db.Float, default=100000)

# ヘルパー関数
def get_jst_now():
    jst = pytz.timezone('Asia/Tokyo')
    return datetime.now(jst)

def get_jst_today():
    return get_jst_now().date()

def generate_order_id():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def generate_barcode(order_id):
    code = barcode.get_barcode_class('code128')
    barcode_instance = code(order_id, writer=ImageWriter())
    buffer = BytesIO()
    barcode_instance.write(buffer)
    buffer.seek(0)
    return buffer

def save_image(file):
    if file and file.filename:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return filename
    return None

def is_order_allowed(product):
    now = get_jst_now()
    if product.deadline_time:
        deadline = datetime.combine(get_jst_today(), product.deadline_time)
        jst = pytz.timezone('Asia/Tokyo')
        deadline = jst.localize(deadline)
        if now > deadline:
            return False
    return True

def get_available_products():
    today = get_jst_today()
    products = Product.query.filter(
        (Product.show_date == today) | (Product.show_date == None)
    ).all()
    return [p for p in products if is_order_allowed(p) and (p.stock_quantity is None or p.stock_quantity > 0)]

def predict_sales(product_id, days_ahead=3):
    sales_data = SalesInput.query.filter_by(product_id=product_id).order_by(SalesInput.sale_date).all()
    if len(sales_data) < 2:
        return []

    dates = [s.sale_date for s in sales_data]
    quantities = [s.quantity_sold for s in sales_data]

    # 日付を数値に変換
    date_nums = [(d - dates[0]).days for d in dates]

    # 線形回帰で予測
    X = np.array(date_nums).reshape(-1, 1)
    y = np.array(quantities)

    model = LinearRegression()
    model.fit(X, y)

    # 予測
    predictions = []
    last_date = dates[-1]
    for i in range(1, days_ahead + 1):
        future_date = last_date + timedelta(days=i)
        future_date_num = (future_date - dates[0]).days
        prediction = model.predict([[future_date_num]])[0]
        predictions.append({
            'date': future_date,
            'predicted_quantity': max(0, int(prediction))
        })

    return predictions

# データベース初期化
def create_tables():
    with app.app_context():
        db.create_all()

        # 管理者アカウントがない場合は作成
        if not Admin.query.first():
            admin = Admin(username='admin', password='admin')
            db.session.add(admin)
            db.session.commit()

        # アプリ設定がない場合は作成
        if not AppConfig.query.first():
            config = AppConfig()
            db.session.add(config)
            db.session.commit()

# ------------------ 利用者向け ------------------

@app.route('/')
def index():
    return redirect(url_for('user_top'))

@app.route('/user_top')
def user_top():
    products = get_available_products()
    categories = {}
    for product in products:
        if product.category not in categories:
            categories[product.category] = []
        categories[product.category].append(product)

    return render_template('user_top.html', categories=categories)

@app.route('/order', methods=['GET', 'POST'])
def order():
    if request.method == 'POST':
        # 金額制限チェック
        total_price = float(request.form['total_price'])
        config = AppConfig.query.first()
        max_amount = config.max_order_amount if config else 100000

        if total_price > max_amount:
            flash(f'注文金額が上限（{max_amount:,.0f}円）を超えています', 'error')
            products = get_available_products()
            categories = {}
            for product in products:
                if product.category not in categories:
                    categories[product.category] = []
                categories[product.category].append(product)
            return render_template('order.html', categories=categories, config=config)

        # 在庫チェック
        stock_errors = []
        for key, value in request.form.items():
            if key.startswith('quantity_') and int(value) > 0:
                product_id = int(key.replace('quantity_', ''))
                product = Product.query.get(product_id)
                if product and product.stock_quantity is not None:
                    if product.stock_quantity < int(value):
                        stock_errors.append(f'{product.name}: 在庫不足（残り{product.stock_quantity}個）')

        if stock_errors:
            for error in stock_errors:
                flash(error, 'error')
            products = get_available_products()
            categories = {}
            for product in products:
                if product.category not in categories:
                    categories[product.category] = []
                categories[product.category].append(product)
            return render_template('order.html', categories=categories, config=config)

        # 注文処理
        order_id = generate_order_id()
        today = get_jst_today()

        new_order = Order(
            order_id=order_id,
            customer_name=request.form['name'],
            phone_number=request.form.get('phone', ''),
            grade=int(request.form.get('grade', 0)) if request.form.get('grade') else None,
            class_num=int(request.form.get('class', 0)) if request.form.get('class') else None,
            number=int(request.form.get('number', 0)) if request.form.get('number') else None,
            total_price=total_price,
            order_date=today,
            created_at=get_jst_now()
        )

        db.session.add(new_order)
        db.session.flush()  # IDを取得するため

        # 注文アイテムの処理
        for key, value in request.form.items():
            if key.startswith('quantity_') and int(value) > 0:
                product_id = int(key.replace('quantity_', ''))
                product = Product.query.get(product_id)
                if product:
                    order_item = OrderItem(
                        order_id=new_order.id,
                        product_id=product_id,
                        quantity=int(value),
                        price=product.price
                    )
                    db.session.add(order_item)

                    # 在庫更新（在庫数が設定されている場合のみ）
                    if product.stock_quantity is not None:
                        product.stock_quantity -= int(value)

                    # 販売数を自動記録
                    sales_record = SalesInput.query.filter_by(
                        product_id=product_id, 
                        sale_date=today
                    ).first()

                    if sales_record:
                        sales_record.quantity_sold += int(value)
                    else:
                        sales_record = SalesInput(
                            product_id=product_id,
                            sale_date=today,
                            quantity_sold=int(value)
                        )
                        db.session.add(sales_record)

        db.session.commit()
        session['last_order_id'] = order_id
        return redirect('/order_done')

    products = get_available_products()
    categories = {}
    for product in products:
        if product.category not in categories:
            categories[product.category] = []
        categories[product.category].append(product)

    config = AppConfig.query.first()
    return render_template('order.html', categories=categories, config=config)

@app.route('/order_done')
def order_done():
    order_id = session.get('last_order_id')
    if not order_id:
        return redirect('/user_top')

    order = Order.query.filter_by(order_id=order_id).first()
    return render_template('done.html', order=order)

@app.route('/barcode/<order_id>')
def barcode_image(order_id):
    buffer = generate_barcode(order_id)
    return send_file(buffer, mimetype='image/png')

# ------------------ 管理者向け ------------------

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        admin = Admin.query.filter_by(username=request.form['username']).first()
        if admin and admin.password == request.form['password']:
            session['admin'] = True
            return redirect('/admin_top')
        flash('ログイン情報が間違っています', 'error')
    return render_template('admin_login.html')

@app.route('/admin_top')
def admin_top():
    if not session.get('admin'):
        return redirect('/admin_login')

    today = get_jst_today()
    today_orders = Order.query.filter_by(order_date=today).count()
    total_sales = db.session.query(db.func.sum(Order.total_price)).filter_by(order_date=today).scalar() or 0

    return render_template('admin_top.html', 
                         today_orders=today_orders, 
                         total_sales=total_sales,
                         today=today)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/user_top')

@app.route('/product_register', methods=['GET', 'POST'])
def product_register():
    if not session.get('admin'):
        return redirect('/admin_login')

    if request.method == 'POST':
        product_id = request.form.get('product_id')
        name = request.form['name']
        category = request.form.get('category')
        new_category = request.form.get('new_category')

        # 新しいカテゴリが入力されている場合はそれを使用
        if new_category and new_category.strip():
            category = new_category.strip()

        # カテゴリが空の場合はエラーメッセージを表示
        if not category or category.strip() == '':
            flash('カテゴリを選択または入力してください', 'error')
            products = Product.query.all()
            categories = list(set([p.category for p in products]))
            return render_template('product_register.html', products=products, categories=categories)

        price = float(request.form['price'])
        stock_quantity = int(request.form['stock_quantity']) if request.form.get('stock_quantity') else None
        deadline_time = datetime.strptime(request.form['deadline_time'], '%H:%M').time() if request.form['deadline_time'] else None
        show_date = datetime.strptime(request.form['show_date'], '%Y-%m-%d').date() if request.form.get('show_date') else None

        img_filename = save_image(request.files.get('image'))

        if product_id:
            product = Product.query.get(product_id)
            product.name = name
            product.category = category
            product.price = price
            product.stock_quantity = stock_quantity
            product.deadline_time = deadline_time
            product.show_date = show_date
            if img_filename:
                product.image = img_filename
        else:
            product = Product(
                name=name,
                category=category,
                price=price,
                stock_quantity=stock_quantity,
                deadline_time=deadline_time,
                show_date=show_date,
                image=img_filename
            )
            db.session.add(product)

        db.session.commit()
        return redirect('/product_register')

    products = Product.query.all()
    categories = list(set([p.category for p in products]))
    return render_template('product_register.html', products=products, categories=categories)

@app.route('/order_management')
def order_management():
    if not session.get('admin'):
        return redirect('/admin_login')

    date_str = request.args.get('date', get_jst_today().isoformat())
    search_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    search_id = request.args.get('search_id', '')

    query = Order.query.filter_by(order_date=search_date)
    if search_id:
        query = query.filter(Order.order_id.like(f'%{search_id}%'))

    orders = query.all()

    # 商品別集計
    product_counts = {}
    for order in orders:
        for item in order.items:
            if item.product.name not in product_counts:
                product_counts[item.product.name] = 0
            product_counts[item.product.name] += item.quantity

    return render_template('order_management.html', 
                         orders=orders, 
                         product_counts=product_counts,
                         search_date=search_date,
                         search_id=search_id)

@app.route('/update_order_status/<int:order_id>/<status>')
def update_order_status(order_id, status):
    if not session.get('admin'):
        return redirect('/admin_login')

    order = Order.query.get(order_id)
    if order:
        order.status = status
        db.session.commit()

    return redirect(url_for('order_management'))


@app.route('/delete_order/<int:order_id>')
def delete_order(order_id):
    if not session.get('admin'):
        return redirect('/admin_login')

    order = Order.query.get(order_id)
    if order:
        db.session.delete(order)
        db.session.commit()

    return redirect(url_for('order_management'))

@app.route('/export_csv')
def export_csv():
    if not session.get('admin'):
        return redirect('/admin_login')

    date_str = request.args.get('date', get_jst_today().isoformat())
    search_date = datetime.strptime(date_str, '%Y-%m-%d').date()

    orders = Order.query.filter_by(order_date=search_date).all()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['注文ID', '顧客名', '電話番号', '学年', '組', '番号', '商品名', '個数', '単価', '小計', '合計金額', 'ステータス'])

    for order in orders:
        for item in order.items:
            writer.writerow([
                order.order_id,
                order.customer_name,
                order.phone_number or '',
                order.grade or '',
                order.class_num or '',
                order.number or '',
                item.product.name,
                item.quantity,
                item.price,
                item.quantity * item.price,
                order.total_price,
                order.status
            ])

    output.seek(0)
    buffer = BytesIO()
    buffer.write(output.getvalue().encode('utf-8'))
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'orders_{date_str}.csv'
    )

@app.route('/sales_input', methods=['GET', 'POST'])
def sales_input():
    if not session.get('admin'):
        return redirect('/admin_login')

    if request.method == 'POST':
        product_id = int(request.form['product_id'])
        sale_date = datetime.strptime(request.form['sale_date'], '%Y-%m-%d').date()
        quantity_sold = int(request.form['quantity_sold'])

        # 既存データをチェック
        existing = SalesInput.query.filter_by(product_id=product_id, sale_date=sale_date).first()
        if existing:
            existing.quantity_sold = quantity_sold
        else:
            sales_record = SalesInput(
                product_id=product_id,
                sale_date=sale_date,
                quantity_sold=quantity_sold
            )
            db.session.add(sales_record)

        db.session.commit()
        return redirect('/sales_input')

    products = Product.query.all()
    sales_records = SalesInput.query.order_by(SalesInput.sale_date.desc()).limit(20).all()

    return render_template('sales_input.html', products=products, sales_records=sales_records)

@app.route('/sales_prediction')
def sales_prediction():
    if not session.get('admin'):
        return redirect('/admin_login')

    products = Product.query.all()
    prediction_data = []

    for product in products:
        predictions = predict_sales(product.id)
        if predictions:
            prediction_data.append({
                'product': product,
                'predictions': predictions
            })

    # グラフデータ作成
    if prediction_data:
        fig = go.Figure()

        for item in prediction_data:
            dates = [p['date'] for p in item['predictions']]
            quantities = [p['predicted_quantity'] for p in item['predictions']]

            fig.add_trace(go.Scatter(
                x=dates,
                y=quantities,
                mode='lines+markers',
                name=item['product'].name
            ))

        fig.update_layout(
            title='販売予測（3日間）',
            xaxis_title='日付',
            yaxis_title='予測販売数',
            hovermode='x unified'
        )

        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        graph_json = None

    return render_template('sales_prediction.html', 
                         prediction_data=prediction_data,
                         graph_json=graph_json)

@app.route('/password_change', methods=['GET', 'POST'])
def password_change():
    if not session.get('admin'):
        return redirect('/admin_login')

    if request.method == 'POST':
        admin = Admin.query.first()
        if admin.password == request.form['current_password']:
            admin.password = request.form['new_password']
            db.session.commit()
            flash('パスワードを変更しました', 'success')
            return redirect('/admin_top')
        flash('現在のパスワードが違います', 'error')

    return render_template('password_change.html')

@app.route('/toggle_product_visibility/<int:product_id>', methods=['POST'])
def toggle_product_visibility(product_id):
    if not session.get('admin'):
        return {'success': False, 'error': 'Unauthorized'}, 401

    product = Product.query.get(product_id)
    if not product:
        return {'success': False, 'error': 'Product not found'}, 404

    # 表示日の切り替え（None = 常時表示、今日の日付 = 今日のみ表示）
    if product.show_date is None:
        product.show_date = get_jst_today()
    else:
        product.show_date = None

    db.session.commit()
    return {'success': True, 'visible': product.show_date is None}

@app.route('/delete_product/<int:product_id>')
def delete_product(product_id):
    if not session.get('admin'):
        return redirect('/admin_login')

    product = Product.query.get(product_id)
    if product:
        db.session.delete(product)
        db.session.commit()
        flash('商品を削除しました', 'success')

    return redirect('/product_register')

@app.route('/app_config', methods=['GET', 'POST'])
def app_config():
    if not session.get('admin'):
        return redirect('/admin_login')

    if request.method == 'POST':
        config = AppConfig.query.first()
        if not config:
            config = AppConfig()
            db.session.add(config)

        config.require_phone = 'require_phone' in request.form
        config.require_grade = 'require_grade' in request.form
        config.require_class = 'require_class' in request.form
        config.require_number = 'require_number' in request.form
        config.max_order_amount = float(request.form.get('max_order_amount', 100000))

        db.session.commit()
        flash('設定を更新しました', 'success')
        return redirect('/app_config')

    config = AppConfig.query.first()
    if not config:
        config = AppConfig()
        db.session.add(config)
        db.session.commit()

    return render_template('app_config.html', config=config)

@app.route('/delete_sales_record/<int:record_id>')
def delete_sales_record(record_id):
    if not session.get('admin'):
        return redirect('/admin_login')

    record = SalesInput.query.get(record_id)
    if record:
        db.session.delete(record)
        db.session.commit()
        flash('販売記録を削除しました', 'success')

    return redirect('/sales_input')

@app.route('/delete_sales_by_date', methods=['POST'])
def delete_sales_by_date():
    if not session.get('admin'):
        return redirect('/admin_login')

    date_str = request.form.get('delete_date')
    if date_str:
        delete_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        records = SalesInput.query.filter_by(sale_date=delete_date).all()
        for record in records:
            db.session.delete(record)
        db.session.commit()
        flash(f'{delete_date}の販売記録を削除しました', 'success')

    return redirect('/sales_input')

if __name__ == '__main__':
    create_tables()
    app.run(host='0.0.0.0', port=5000, debug=True)