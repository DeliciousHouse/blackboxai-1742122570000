# Home Assistant 3D Blueprint Generator

Generate dynamic 3D home blueprints from Bluetooth sensor data in Home Assistant. This add-on processes Bluetooth signal strengths to create and maintain an accurate spatial map of your home.

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2FDeliciousHouse%2FHome-Assistant_3D_Blueprint_Generator)

## Features

- **Dynamic Blueprint Generation**: Automatically creates 3D home blueprints from Bluetooth sensor data
- **Real-time Processing**: Continuously updates spatial mapping based on sensor readings
- **Interactive UI**: Modern web interface for viewing and adjusting blueprints
- **Local Processing**: All data processing happens locally for maximum privacy
- **Home Assistant Integration**: Direct integration with Home Assistant's Bluetooth devices
- **Manual Adjustments**: Interface for fine-tuning room layouts and dimensions

## Quick Start

1. Click the "Add to Home Assistant" button above
2. Click "Install" in the Home Assistant Add-on Store
3. Configure the required settings
4. Start the add-on
5. Click "OPEN WEB UI" to access the interface

## Documentation

- [Installation Guide](blueprint_generator/DOCS.md#installation)
- [Configuration](blueprint_generator/DOCS.md#configuration)
- [Usage Guide](blueprint_generator/DOCS.md#usage)
- [Troubleshooting](blueprint_generator/DOCS.md#troubleshooting)
- [Contributing](CONTRIBUTING.md)

## Support

- [Open an issue](https://github.com/DeliciousHouse/Home-Assistant_3D_Blueprint_Generator/issues)
- [Discord Chat](https://discord.gg/c5DvZ4e)
- [Home Assistant Community](https://community.home-assistant.io)

## Development

### Prerequisites

- Docker
- Python 3.9 or higher
- Git

### Local Testing

```bash
cd blueprint_generator
./test_locally.sh
```

### Running Tests

```bash
pytest                 # Run all tests
pytest -v             # Verbose output
pytest --cov         # With coverage report
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Home Assistant community
- Three.js for 3D rendering
- Contributors and testers

## Security

- All data processed locally
- No external API calls
- Database access restricted
- Input validation on all endpoints
- Secure password storage
