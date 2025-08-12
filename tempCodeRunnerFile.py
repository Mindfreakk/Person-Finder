@app.route('/search', methods=['GET', 'POST'])
def search():
    matches = []
    photo_rel_path = None

    if request.method == 'POST':
        photo = request.files.get('photo')
        if not photo or photo.filename == '':
            flash('No photo uploaded', 'error')
            return render_template('search.html', matches=None, photo_path=None)

        unique_filename = f"{uuid.uuid4().hex}_{secure_filename(photo.filename)}"
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        photo_rel_path = os.path.join('uploads', unique_filename).replace("\\", "/")  # For preview
        photo.save(photo_path)

        try:
            search_encoding = extract_face_encoding(photo_path)
        except ValueError as e:
            os.remove(photo_path)
            flash(str(e), 'error')
            return render_template('search.html', matches=None, photo_path=None)

        registered_people = get_all_registered_people()
        for person in registered_people:
            registered_encoding = np.array(person['face_encoding'])
            if compare_faces(search_encoding, registered_encoding):
                relative_photo_path = os.path.relpath(person['photo_path'], 'static')
                person['photo_path'] = relative_photo_path.replace("\\", "/")
                matches.append(person)

        if matches:
            flash(f'Found {len(matches)} matching person(s).', 'success')
        else:
            flash('No matching person found.', 'error')

        try:
            os.remove(photo_path)
        except Exception as e:
            print(f"Warning: could not delete search photo: {e}")

        return render_template('search.html', matches=matches, photo_path=photo_rel_path)

    return render_template('search.html', matches=None, photo_path=None)