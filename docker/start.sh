#!/bin/bash

echo "🚀 Starting YuLan-OneSim container..."

# Create necessary directories
mkdir -p /app/logs /var/log/supervisor

# Set permissions
chown -R www-data:www-data /var/www/html
chmod -R 755 /var/www/html

# Check configuration file
if [ ! -f "/app/config/config.json" ]; then
    echo "⚠️  Configuration file does not exist, using default configuration"
    # Default configuration file can be created here
fi

# Display startup information
echo "🌐 YuLan-OneSim is starting..."
echo "📍 Web interface: http://localhost"
echo "📍 API documentation: http://localhost/docs"
echo "📍 CLI tool: docker exec -it <container_id> yulan-onesim-cli --help"

# Start supervisor to manage all services
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf