[app]
title = Face Finder
package.name = facefinder
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 1.0
requirements = python3,kivy,opencv-python,requests
orientation = portrait
fullscreen = 1
android.permissions = CAMERA,INTERNET

[buildozer]
log_level = 2
warn_on_root = 1
