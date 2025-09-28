self.addEventListener('push', event => {
  let data = { title: 'Notification', body: '', url: '/' };

  try {
    data = event.data ? event.data.json() : data;
  } catch (err) {
    console.error('Push event data parsing error:', err);
  }

  const options = {
    body: data.body || '',
    icon: data.icon || '/static/Images/notification_icon.png',
    badge: data.badge || '/static/Images/icon-192.png',
    vibrate: [100, 50, 100],
    data: { url: data.url || '/' },
    timestamp: Date.now()
  };

  event.waitUntil(self.registration.showNotification(data.title || 'Notification', options));
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clientList => {
      for (const client of clientList) {
        if ('focus' in client) return client.focus();
      }
      return clients.openWindow(event.notification.data.url || '/');
    }).catch(err => console.error('Notification click error:', err))
  );
});
