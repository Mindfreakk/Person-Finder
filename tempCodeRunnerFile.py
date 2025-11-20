# ---- Dashboard role badge for logged-in user (Super Admin vs Admin) ----
    # This email is your *primary* Super Admin. Set in env, e.g. ADMIN_EMAIL=ammehz09@gmail.com
    super_admin_email = (os.getenv("ADMIN_EMAIL") or "").strip().lower()

    admin_user = (
    db.session.get(User, int(current_user.id))
    if current_user.is_authenticated
    else None
)


    is_super_admin_current = bool(
        admin_user
        and super_admin_email
        and (admin_user.email or "").strip().lower() == super_admin_email
    )

    if is_super_admin_current:
        dashboard_role_label = "Super Admin"
        dashboard_role_slug = "super-admin"
        dashboard_role_icon = "👑"
    else:
        # Normal admin (users cannot access this dashboard otherwise)
        dashboard_role_label = "Admin"
        dashboard_role_slug = "admin"
        dashboard_role_icon = "🛡️"

    # lowercase display username for the header
    display_username = ""
    if admin_user and getattr(admin_user, "username", None):
        display_username = (admin_user.username or "").strip().lower()

    return render_template(
        "admin_dashboard.html",
        # people
        people=people,
        page=page,
        total_pages=total_pages,
        # users
        users=users,
        # logs
        search_logs=search_logs,
        # stats/system
        stats=stats,
        app_version=app_version,
        recent_events=recent_events,
        # feedback
        feedback_total=feedback_total,
        feedback_avg_rating=feedback_avg_rating,
        feedback_negative_count=feedback_negative_count,
        feedback_hist=feedback_hist,
        recent_feedback=recent_feedback,
        # dashboard greeting + role pill
        dashboard_role_label=dashboard_role_label,
        dashboard_role_slug=dashboard_role_slug,
        dashboard_role_icon=dashboard_role_icon,
        display_username=display_username,
        # expose super admin email for the template logic you already have
        super_admin_email=super_admin_email,
        primary_super_admin_email=super_admin_email,
    )