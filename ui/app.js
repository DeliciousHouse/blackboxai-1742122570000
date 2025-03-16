// Global variables for Three.js
let scene, camera, renderer, controls;
let blueprint = null;
let selectedRoom = null;
let selectedWall = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    initEventListeners();
    fetchBlueprint();
});

// Initialize Three.js scene
function initThreeJS() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf3f4f6);

    // Create camera
    camera = new THREE.PerspectiveCamera(
        75,
        document.getElementById('viewport').clientWidth / document.getElementById('viewport').clientHeight,
        0.1,
        1000
    );
    camera.position.set(10, 10, 10);
    camera.lookAt(0, 0, 0);

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(
        document.getElementById('viewport').clientWidth,
        document.getElementById('viewport').clientHeight
    );
    document.getElementById('viewport').appendChild(renderer.domElement);

    // Add orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Add grid helper
    const gridHelper = new THREE.GridHelper(20, 20);
    scene.add(gridHelper);

    // Start animation loop
    animate();
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Initialize event listeners
function initEventListeners() {
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', fetchBlueprint);

    // Camera control buttons
    document.getElementById('rotateLeftBtn').addEventListener('click', () => {
        controls.rotateLeft(Math.PI / 4);
    });

    document.getElementById('rotateRightBtn').addEventListener('click', () => {
        controls.rotateLeft(-Math.PI / 4);
    });

    document.getElementById('zoomInBtn').addEventListener('click', () => {
        camera.position.multiplyScalar(0.8);
    });

    document.getElementById('zoomOutBtn').addEventListener('click', () => {
        camera.position.multiplyScalar(1.2);
    });

    document.getElementById('resetViewBtn').addEventListener('click', resetCamera);

    // Update form
    document.getElementById('updateForm').addEventListener('submit', handleFormSubmit);

    // Add room button
    document.getElementById('addRoomBtn').addEventListener('click', handleAddRoom);

    // Window resize handler
    window.addEventListener('resize', handleResize);
}

// Fetch blueprint from API
async function fetchBlueprint() {
    showLoader();
    try {
        const response = await fetch('/api/blueprint');
        if (!response.ok) throw new Error('Failed to fetch blueprint');
        
        blueprint = await response.json();
        updateScene();
        updateUI();
        showMessage('Blueprint updated successfully', 'success');
    } catch (error) {
        console.error('Error fetching blueprint:', error);
        showMessage('Failed to fetch blueprint', 'error');
    } finally {
        hideLoader();
    }
}

// Update Three.js scene with blueprint data
function updateScene() {
    // Clear existing objects
    while(scene.children.length > 0) {
        scene.remove(scene.children[0]);
    }

    // Add grid and lights back
    const gridHelper = new THREE.GridHelper(20, 20);
    scene.add(gridHelper);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    if (!blueprint) return;

    // Add rooms
    blueprint.rooms.forEach(room => {
        const roomGeometry = createRoomGeometry(room);
        const roomMaterial = new THREE.MeshPhongMaterial({
            color: getRoomColor(room.type),
            transparent: true,
            opacity: 0.8,
            side: THREE.DoubleSide
        });
        const roomMesh = new THREE.Mesh(roomGeometry, roomMaterial);
        roomMesh.userData = { type: 'room', id: room.id };
        scene.add(roomMesh);
    });

    // Add walls
    blueprint.walls.forEach(wall => {
        const wallGeometry = createWallGeometry(wall);
        const wallMaterial = new THREE.MeshPhongMaterial({
            color: 0x808080,
            transparent: false
        });
        const wallMesh = new THREE.Mesh(wallGeometry, wallMaterial);
        wallMesh.userData = { type: 'wall', id: wall.id };
        scene.add(wallMesh);
    });
}

// Create room geometry from vertices
function createRoomGeometry(room) {
    const shape = new THREE.Shape();
    const vertices = room.vertices;
    
    shape.moveTo(vertices[0][0], vertices[0][1]);
    for (let i = 1; i < vertices.length; i++) {
        shape.lineTo(vertices[i][0], vertices[i][1]);
    }
    shape.lineTo(vertices[0][0], vertices[0][1]);

    const geometry = new THREE.ExtrudeGeometry(shape, {
        depth: room.height,
        bevelEnabled: false
    });

    return geometry;
}

// Create wall geometry
function createWallGeometry(wall) {
    const wallShape = new THREE.Shape();
    wallShape.moveTo(wall.start[0], wall.start[1]);
    wallShape.lineTo(wall.end[0], wall.end[1]);

    const extrudeSettings = {
        steps: 1,
        depth: wall.height,
        bevelEnabled: false
    };

    return new THREE.ExtrudeGeometry(wallShape, extrudeSettings);
}

// Get color for room type
function getRoomColor(type) {
    const colors = {
        LIVING: 0x90cdf4,
        BEDROOM: 0xfbd5e0,
        KITCHEN: 0x9ae6b4,
        BATHROOM: 0xe9d8fd,
        HALLWAY: 0xfbd38d
    };
    return colors[type] || 0xcccccc;
}

// Update UI elements with blueprint data
function updateUI() {
    const roomList = document.getElementById('roomList');
    const wallList = document.getElementById('wallList');

    // Clear existing lists
    roomList.innerHTML = '';
    wallList.innerHTML = '';

    // Update room list
    blueprint.rooms.forEach(room => {
        const roomItem = document.createElement('div');
        roomItem.className = `room-item p-2 border rounded-lg mb-2 cursor-pointer 
            ${selectedRoom?.id === room.id ? 'selected' : ''}`;
        roomItem.innerHTML = `
            <div class="flex justify-between items-center">
                <span class="font-medium">${room.id}</span>
                <span class="room-type-badge ${room.type}">${room.type}</span>
            </div>
            <div class="text-sm text-gray-500">
                Area: ${room.area.toFixed(2)} m²
            </div>
        `;
        roomItem.addEventListener('click', () => selectRoom(room));
        roomList.appendChild(roomItem);
    });

    // Update wall list
    blueprint.walls.forEach(wall => {
        const wallItem = document.createElement('div');
        wallItem.className = `wall-item p-2 border rounded-lg mb-2 cursor-pointer
            ${selectedWall?.id === wall.id ? 'selected' : ''}`;
        wallItem.innerHTML = `
            <div class="flex justify-between items-center">
                <span class="font-medium">${wall.id}</span>
                <span class="text-sm text-gray-500">
                    ${wall.room1_id} - ${wall.room2_id}
                </span>
            </div>
        `;
        wallItem.addEventListener('click', () => selectWall(wall));
        wallList.appendChild(wallItem);
    });
}

// Handle room selection
function selectRoom(room) {
    selectedRoom = room;
    selectedWall = null;
    updateUI();
    
    // Update form
    document.getElementById('roomName').value = room.id;
    document.getElementById('roomType').value = room.type;
}

// Handle wall selection
function selectWall(wall) {
    selectedWall = wall;
    selectedRoom = null;
    updateUI();
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    if (!selectedRoom) return;

    showLoader();
    try {
        const updatedRoom = {
            ...selectedRoom,
            id: document.getElementById('roomName').value,
            type: document.getElementById('roomType').value
        };

        const response = await fetch('/api/blueprint/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                rooms: [updatedRoom],
                walls: blueprint.walls
            })
        });

        if (!response.ok) throw new Error('Failed to update blueprint');

        blueprint = await response.json();
        updateScene();
        updateUI();
        showMessage('Blueprint updated successfully', 'success');
    } catch (error) {
        console.error('Error updating blueprint:', error);
        showMessage('Failed to update blueprint', 'error');
    } finally {
        hideLoader();
    }
}

// Handle adding a new room
function handleAddRoom() {
    // Implementation for adding a new room
    showMessage('Room addition not implemented yet', 'error');
}

// Handle window resize
function handleResize() {
    const viewport = document.getElementById('viewport');
    camera.aspect = viewport.clientWidth / viewport.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewport.clientWidth, viewport.clientHeight);
}

// Reset camera position
function resetCamera() {
    camera.position.set(10, 10, 10);
    camera.lookAt(0, 0, 0);
    controls.reset();
}

// Utility functions for loader and messages
function showLoader() {
    document.getElementById('loader').classList.add('show');
}

function hideLoader() {
    document.getElementById('loader').classList.remove('show');
}

function showMessage(text, type) {
    const messageEl = document.getElementById('message');
    const messageText = messageEl.querySelector('.message-text');
    
    messageText.textContent = text;
    messageEl.className = `fixed top-4 right-4 z-40 ${type}`;
    messageEl.classList.add('show');
    
    setTimeout(() => {
        messageEl.classList.remove('show');
    }, 3000);
}

// Debounce utility function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
