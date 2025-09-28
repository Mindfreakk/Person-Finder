# --- Register Person ---
@app.route("/register", methods=["GET", "POST"])
def register():
    """
    Improved registration route:
    - Accepts uploaded file 'photo' (or 'photo_file') or base64 in 'photo_input'
    - Validates required fields server-side (including agreement checkbox)
    - If validation fails, re-renders the form with errors + previous values (no redirect),
      and provides 'focus_field' which is the first invalid field id for client-side scrolling.
    - If face detection fails we remove the saved file and return an error.
    """
    filename_for_preview = None
    errors = {}
    form_data = {}     # will hold values to re-populate the form on error
    focus_field = None

    if request.method == "POST":
        try:
            # Grab raw form and files
            raw_form = request.form or {}
            photo_file = request.files.get("photo") or request.files.get("photo_file")
            photo_base64 = (raw_form.get("photo_input") or "").strip()

            # copy values to form_data so template can re-populate fields if validation fails
            keys_to_copy = [
                "full_name","age","gender","guardian_name","phone_number","address","last_seen",
                "registered_by_name","registered_by_phone","registered_by_relation","agreement"
            ]
            for k in keys_to_copy:
                form_data[k] = raw_form.get(k, "") if raw_form.get(k, "") is not None else ""

            # --- Server-side validation: required fields ---
            required_fields = {
                "full_name": "Full name is required.",
                "age": "Age is required.",
                "gender": "Please select a gender.",
                "guardian_name": "Guardian name is required.",
                "phone_number": "Phone number is required.",
                "address": "Address is required.",
                "registered_by_name": "Your name is required.",
                "registered_by_phone": "Your phone number is required.",
                "registered_by_relation": "Relation is required."
            }

            for fid, msg in required_fields.items():
                val = (form_data.get(fid) or "").strip()
                if val == "":
                    errors[fid] = msg
                    if not focus_field:
                        focus_field = fid

            # agreement checkbox must be present and checked (value typically "on")
            agreement_val = form_data.get("agreement", "")
            if not agreement_val or not str(agreement_val).lower() in ("1","true","on","yes"):
                errors["agreement"] = "You must agree to the Privacy Policy and Terms."
                if not focus_field:
                    focus_field = "agreement"

            # Validate photo presence (file or base64)
            if not (photo_file and getattr(photo_file, "filename", "").strip()) and not photo_base64:
                errors["photo"] = "Photo is required (upload or use webcam)."
                if not focus_field:
                    focus_field = "photo"

            photo_abs_path = None
            try:
                # Save uploaded file or decode base64 to file
                if photo_file and getattr(photo_file, "filename", "").strip():
                    # save_uploaded_file should return absolute path to saved file
                    photo_abs_path = save_uploaded_file(photo_file)
                elif photo_base64:
                    # save_base64_image should return absolute path to saved file
                    photo_abs_path = save_base64_image(photo_base64)
            except Exception as e:
                logger.exception("Error saving uploaded photo: %s", e)
                errors["photo"] = "Unable to save uploaded photo."
                if not focus_field:
                    focus_field = "photo"

            # If any validation errors so far -> cleanup saved photo & return page with errors
            if errors:
                # remove saved photo if created
                if photo_abs_path:
                    try:
                        os.remove(photo_abs_path)
                    except Exception:
                        pass
                    photo_abs_path = None
                # prepare preview filename only when there's a valid saved photo
                filename_for_preview = None
                return render_template("register.html", filename=filename_for_preview,
                                       errors=errors, form_data=form_data, focus_field=focus_field), 400

            # At this point photo_abs_path should be present; try face encoding
            try:
                image = face_recognition.load_image_file(photo_abs_path)
                encodings = face_recognition.face_encodings(image)
                if not encodings:
                    # no face detected
                    errors["photo"] = "No face detected in the uploaded photo. Please upload a clear, front-facing image."
                    try:
                        os.remove(photo_abs_path)
                    except Exception:
                        pass
                    photo_abs_path = None
                    if not focus_field:
                        focus_field = "photo"
                    return render_template("register.html", filename=None,
                                           errors=errors, form_data=form_data, focus_field=focus_field), 400
            except Exception as ex:
                logger.exception("Error processing face encoding: %s", ex)
                errors["photo"] = "Unable to process the uploaded photo."
                try:
                    if photo_abs_path:
                        os.remove(photo_abs_path)
                except Exception:
                    pass
                photo_abs_path = None
                if not focus_field:
                    focus_field = "photo"
                return render_template("register.html", filename=None,
                                       errors=errors, form_data=form_data, focus_field=focus_field), 400

            # Prepare person_data (coerce age safely)
            try:
                age_val = None
                if form_data.get("age"):
                    try:
                        age_val = int(form_data.get("age"))
                    except ValueError:
                        age_val = None
                person_data = {
                    "full_name": (form_data.get("full_name") or "").strip(),
                    "age": age_val,
                    "gender": (form_data.get("gender") or "").strip(),
                    "guardian_name": (form_data.get("guardian_name") or "").strip(),
                    "phone_number": (form_data.get("phone_number") or "").strip(),
                    "address": (form_data.get("address") or "").strip(),
                    "last_seen": (form_data.get("last_seen") or "").strip(),
                    "photo_path": os.path.basename(photo_abs_path) if photo_abs_path else None,
                    "face_encoding": encodings[0].tolist() if encodings else None,
                    "created_by": current_user.id if current_user and getattr(current_user, "is_authenticated", False) else None,
                    "registered_by_name": (form_data.get("registered_by_name") or "").strip(),
                    "registered_by_phone": (form_data.get("registered_by_phone") or "").strip(),
                    "registered_by_relation": (form_data.get("registered_by_relation") or "").strip(),
                }
            except Exception as e:
                logger.exception("Error preparing person_data: %s", e)
                # cleanup photo
                try:
                    if photo_abs_path:
                        os.remove(photo_abs_path)
                except Exception:
                    pass
                flash("Server error while preparing submitted data.", "error")
                return render_template("register.html", filename=None,
                                       errors={"server":"Server error"}, form_data=form_data, focus_field=None), 500

            # Save into DB (register_person_to_db should validate too)
            try:
                register_person_to_db(person_data)
            except Exception as e:
                logger.exception("DB error registering person: %s", e)
                # cleanup photo if DB save failed
                try:
                    if photo_abs_path:
                        os.remove(photo_abs_path)
                except Exception:
                    pass
                # If register_person_to_db raised with a user-friendly message, show it
                msg = str(e) if str(e) else "Error saving person to database."
                errors["server"] = msg
                return render_template("register.html", filename=None,
                                       errors=errors, form_data=form_data, focus_field=None), 400

            # Success!
            flash("Person registered successfully!", "success")
            return redirect(url_for("home"))

        except Exception as e:
            # Last-resort catch-all: log, attempt cleanup, show friendly error
            logger.exception("Unhandled error in register(): %s", e)
            try:
                if photo_abs_path:
                    os.remove(photo_abs_path)
            except Exception:
                pass
            flash("An unexpected server error occurred while registering. Please try again.", "error")
            return render_template("register.html", filename=None,
                                   errors={"server":"Unexpected error"}, form_data=form_data, focus_field=None), 500

    # GET
    return render_template("register.html", filename=filename_for_preview, form_data={}, errors={}, focus_field=None)