self.addEventListener('install', event => {
    event.waitUntil(
      caches.open('face-app-v1').then(cache => {
        return cache.addAll([
          '/',
          '/static/style.css',
          '/static/manifest.json',
          '/static/icons/icon-192.png',
          '/static/icons/icon-512.png'
        ]);
      })
    );
  });
  
  self.addEventListener('fetch', event => {
    event.respondWith(
      caches.match(event.request).then(resp => {
        return resp || fetch(event.request);
      })
    );
  });
  