// Three.js scene setup
let scene, camera, renderer, controls;
let rooms = [], walls = [], grid;
let wireframeMode = false;

// Initialize Three.js scene
function initScene() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);

    // Create camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(10, 10, 10);

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(document.getElementById('viewer').clientWidth, document.getElementById('viewer').clientHeight);
    document.getElementById('viewer').appendChild(renderer.domElement);

    // Add controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Add grid
    addGrid();

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(10, 20, 10);
    scene.add(directionalLight);

    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);

    // Start animation loop
    animate();
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Window resize handler
function onWindowResize() {
    const container = document.getElementById('viewer');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// Add grid to scene
function addGrid() {
    if (grid) scene.remove(grid);
    grid = new THREE.GridHelper(20, 20, 0x888888, 0xcccccc);
    scene.add(grid);
}

// Create room mesh
function createRoom(room) {
    const geometry = new THREE.BoxGeometry(
        room.dimensions.width,
        room.dimensions.height,
        room.dimensions.length
    );
    const material = new THREE.MeshPhongMaterial({
        color: 0x93c5fd,
        transparent: true,
        opacity: 0.5,
        wireframe: wireframeMode
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(room.center.x, room.center.y, room.center.z);
    return mesh;
}

// Create wall mesh
function createWall(wall) {
    const geometry = new THREE.BoxGeometry(
        wall.length,
        wall.height,
        wall.thickness
    );
    const material = new THREE.MeshPhongMaterial({
        color: 0x475569,
        wireframe: wireframeMode
    });
    const mesh = new THREE.Mesh(geometry, material);

    // Position and rotate wall
    const midpoint = {
        x: (wall.start.x + wall.end.x) / 2,
        y: wall.height / 2,
        z: (wall.start.y + wall.end.y) / 2
    };
    mesh.position.set(midpoint.x, midpoint.y, midpoint.z);
    mesh.rotation.y = wall.angle;

    return mesh;
}

// Update scene with new blueprint data
function updateScene(blueprint) {
    // Clear existing rooms and walls
    rooms.forEach(room => scene.remove(room));
    walls.forEach(wall => scene.remove(wall));
    rooms = [];
    walls = [];

    // Add new rooms
    blueprint.rooms.forEach(roomData => {
        const room = createRoom(roomData);
        scene.add(room);
        rooms.push(room);
    });

    // Add new walls
    blueprint.walls.forEach(wallData => {
        const wall = createWall(wallData);
        scene.add(wall);
        walls.push(wall);
    });
}

// API functions
async function generateBlueprint() {
    try {
        updateStatus('Generating blueprint...', 'warning');
        const response = await fetch('/api/blueprint/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                time_window: parseInt(document.getElementById('updateInterval').value) * 60
            })
        });

        if (!response.ok) throw new Error('Failed to generate blueprint');

        const blueprint = await response.json();
        updateScene(blueprint);
        updateStatus('Blueprint generated successfully', 'success');
        updateLastUpdate();
    } catch (error) {
        console.error('Error generating blueprint:', error);
        updateStatus('Failed to generate blueprint', 'error');
    }
}

async function saveBlueprint() {
    try {
        updateStatus('Saving blueprint...', 'warning');
        const response = await fetch('/api/blueprint/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rooms: rooms.map(room => ({
                    center: room.position,
                    dimensions: room.geometry.parameters
                })),
                walls: walls.map(wall => ({
                    start: { x: wall.position.x - wall.geometry.parameters.width/2, y: wall.position.z },
                    end: { x: wall.position.x + wall.geometry.parameters.width/2, y: wall.position.z },
                    height: wall.geometry.parameters.height,
                    thickness: wall.geometry.parameters.depth,
                    angle: wall.rotation.y
                }))
            })
        });

        if (!response.ok) throw new Error('Failed to save blueprint');

        updateStatus('Blueprint saved successfully', 'success');
    } catch (error) {
        console.error('Error saving blueprint:', error);
        updateStatus('Failed to save blueprint', 'error');
    }
}

// Check this in your frontend JavaScript file
fetch('/api/blueprint')
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(blueprint => {
    console.log('Loaded blueprint with', blueprint.rooms.length, 'rooms');
    renderBlueprint(blueprint);
  })
  .catch(error => {
    console.error('Error loading blueprint:', error);
    displayErrorMessage('Failed to load blueprint data');
  });

// UI update functions
function updateStatus(message, type = 'info') {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = type;
}

function updateLastUpdate() {
    document.getElementById('lastUpdate').textContent =
        `Last update: ${new Date().toLocaleTimeString()}`;
}

// Event listeners
document.getElementById('generateBtn').addEventListener('click', generateBlueprint);
document.getElementById('saveBtn').addEventListener('click', saveBlueprint);
document.getElementById('resetViewBtn').addEventListener('click', () => {
    camera.position.set(10, 10, 10);
    camera.lookAt(0, 0, 0);
});
document.getElementById('toggleWireframeBtn').addEventListener('click', () => {
    wireframeMode = !wireframeMode;
    rooms.forEach(room => room.material.wireframe = wireframeMode);
    walls.forEach(wall => wall.material.wireframe = wireframeMode);
});
document.getElementById('toggleGridBtn').addEventListener('click', () => {
    grid.visible = !grid.visible;
});

// Initialize scene on load
window.addEventListener('load', initScene);
