# ----------------- Utility Function -----------------
def get_stats():
    try:
        registrations = Person.query.count()
        searches = SearchLog.query.count()
        searches_traced = SearchLog.query.filter(SearchLog.success == 1).count()
    except Exception:
        registrations = searches = searches_traced = 0

    return {
        "registrations": registrations,
        "searches": searches,
        "searches_traced": searches_traced,
    }

# ----------------- Home Route -----------------
@app.route("/")
def home():
    stats = get_stats()
    return render_template("home.html", stats=stats)

# ----------------- API Route -----------------
@app.route("/api/stats")
def api_stats():
    stats = get_stats()
    return jsonify(stats)

# ----------------- Auth Routes -----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid username or password.", "error")
            return redirect(url_for("login"))
        login_user(UserLogin(user))
        flash("Logged in successfully!", "success")
        return redirect(url_for("admin_dashboard") if user.role == "admin" else url_for("home"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

# ----------------- Auto-detect LAN IP & build external URLs -----------------
import socket
from urllib.parse import urljoin

def get_local_ip() -> str:
    """Return LAN IP (e.g. 192.168.x.x). Falls back to 127.0.0.1."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # only to learn the outbound iface
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def build_external_url(endpoint: str, **values) -> str:
    """
    Reset links that open on phone:
    - Use PUBLIC_BASE_URL if set (e.g. https://your-domain.com)
    - Else use auto-detected LAN IP + PORT
    """
    base = (os.getenv("PUBLIC_BASE_URL") or "").strip()
    if not base:
        local_ip = get_local_ip()
        port = os.getenv("PORT", "5001")
        base = f"http://{local_ip}:{port}"
    rel = url_for(endpoint, _external=False, **values).lstrip("/")
    return urljoin(base.rstrip("/") + "/", rel)