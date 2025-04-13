document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const startButton = document.getElementById('startButton');
    const playAgainButton = document.getElementById('playAgainButton');
    const timerDisplay = document.getElementById('timer');
    const scoreDisplay = document.getElementById('score');
    const gameOverPanel = document.getElementById('gameOver');
    const finalScoreDisplay = document.getElementById('finalScore');
    const starsCollectedDisplay = document.getElementById('starsCollected');

    // Game state
    let gameRunning = false;
    let gameTime = 60; // 60 seconds game
    let score = 0;
    let starsCollected = 0;
    let timer;
    
    // Critical data metrics
    let totalReactionTimes = []; // Store all reaction times
    let totalObjects = 0; // Total objects spawned
    let successfulActions = 0; // Successful dodges and collections
    let distractionTimes = []; // Times when distractions occur
    let postDistractionReactionTimes = []; // Reaction times within 2s after distractions
    let postDistractionSuccesses = 0; // Successful actions within 2s after distractions
    let postDistractionAttempts = 0; // Total actions attempted within 2s after distractions
    let positiveEmojisCollected = 0; // Happy emojis collected
    let negativeEmojisCollected = 0; // Sad emojis collected
    let blocksHit = 0; // Number of blocks hit
    let blocksDodged = 0; // Number of blocks dodged
    
    // Metrics for data collection
    let reactionTimes = [];
    let obstacleAvoidances = 0;
    let missedRewards = 0;
    let lastDistractionTime = 0;
    let preDistractionSpeed = 0;
    let postDistractionSpeed = 0;
    let distractionEvents = [];
    let lastMouseMoveTime = 0;
    let lastMousePos = { x: 0 };
    let movementSpeeds = [];
    
    // New metrics for Phase 3
    let movementDirectionChanges = 0;
    let hesitations = 0;
    let lastMovementDirection = null; // 'left' or 'right'
    let playerPositions = []; // Record player positions over time
    let idlePeriods = []; // Periods where player doesn't move
    let lastMovementTime = Date.now();
    let emotionalStimulusResponses = []; // Reactions to different emoji types
    
    // Player object
    const player = {
        x: canvas.width / 2,
        y: canvas.height - 50,
        radius: 15,
        color: 'blue',
        speed: 5,
        dragging: false
    };
    
    // Arrays to hold game objects
    let obstacles = [];  // Red blocks to avoid
    let rewards = [];    // Green stars to collect
    let emojis = [];     // Emoji distractions
    let flashActive = false;
    
    // Emoji types
    const emojiTypes = [
        { type: 'happy', value: 2, color: 'yellow', emoji: '😊' },
        { type: 'sad', value: -1, color: 'lightblue', emoji: '😢' },
        { type: 'angry', value: -2, color: 'orange', emoji: '😠' },
        { type: 'neutral', value: 0, color: 'lightgray', emoji: '😐' }
    ];

    // Game parameters
    const obstacleSpeed = 3;
    const rewardSpeed = 2;
    const emojiSpeed = 2.5;
    const obstacleFrequency = 50; // Lower means more frequent
    const rewardFrequency = 150;  // Lower means more frequent
    const emojiFrequency = 200;   // Lower means more frequent
    const flashFrequency = 15000; // Milliseconds between flashes (15 seconds)
    const flashDuration = 500;    // Milliseconds flash stays on screen (0.5 seconds)
    
    // Event listeners for player movement
    canvas.addEventListener('mousedown', function(e) {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        // Check if mouse is on player
        const dx = mouseX - player.x;
        const dy = mouseY - player.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < player.radius) {
            player.dragging = true;
            lastMousePos.x = mouseX;
            lastMouseMoveTime = Date.now();
        }
    });
    
    canvas.addEventListener('mousemove', function(e) {
        if (player.dragging && gameRunning) {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            
            // Calculate movement speed
            const now = Date.now();
            const deltaTime = now - lastMouseMoveTime;
            if (deltaTime > 0) {
                const deltaX = Math.abs(mouseX - lastMousePos.x);
                const speed = deltaX / deltaTime; // pixels per millisecond
                movementSpeeds.push(speed);
                
                // Keep only the last 10 speed measurements
                if (movementSpeeds.length > 10) {
                    movementSpeeds.shift();
                }
                
                // Track direction changes
                const currentDirection = mouseX > lastMousePos.x ? 'right' : mouseX < lastMousePos.x ? 'left' : null;
                if (currentDirection && lastMovementDirection && currentDirection !== lastMovementDirection) {
                    movementDirectionChanges++;
                }
                lastMovementDirection = currentDirection;
                
                // Reset idle timer on movement
                if (deltaX > 1) { // Threshold to consider actual movement
                    // If there was an idle period, record it
                    const idleDuration = now - lastMovementTime;
                    if (idleDuration > 500) { // Idle threshold: 500ms
                        idlePeriods.push({
                            duration: idleDuration,
                            startTime: lastMovementTime,
                            endTime: now,
                            timeSinceLastDistraction: lastMovementTime - lastDistractionTime
                        });
                        
                        // Count as hesitation if close to a distraction
                        if (now - lastDistractionTime < 3000) { // Within 3 seconds of distraction
                            hesitations++;
                        }
                    }
                    lastMovementTime = now;
                }
            }
            
            // Record position for path analysis
            playerPositions.push({
                x: mouseX,
                time: now,
                timeSinceDistraction: now - lastDistractionTime
            });
            
            // Limit the size of position history
            if (playerPositions.length > 100) {
                playerPositions.shift();
            }
            
            lastMousePos.x = mouseX;
            lastMouseMoveTime = now;
            
            player.x = mouseX;
            
            // Keep player within canvas bounds
            if (player.x < player.radius) {
                player.x = player.radius;
            } else if (player.x > canvas.width - player.radius) {
                player.x = canvas.width - player.radius;
            }
        }
    });
    
    canvas.addEventListener('mouseup', function() {
        player.dragging = false;
    });
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', function(e) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const touchX = e.touches[0].clientX - rect.left;
        const touchY = e.touches[0].clientY - rect.top;
        
        const dx = touchX - player.x;
        const dy = touchY - player.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < player.radius) {
            player.dragging = true;
            lastMousePos.x = touchX;
            lastMouseMoveTime = Date.now();
        }
    });
    
    canvas.addEventListener('touchmove', function(e) {
        e.preventDefault();
        if (gameRunning) {
            const rect = canvas.getBoundingClientRect();
            const touchX = e.touches[0].clientX - rect.left;
            
            // Calculate movement speed
            const now = Date.now();
            const deltaTime = now - lastMouseMoveTime;
            if (deltaTime > 0) {
                const deltaX = Math.abs(touchX - lastMousePos.x);
                const speed = deltaX / deltaTime; // pixels per millisecond
                movementSpeeds.push(speed);
                
                // Keep only the last 10 speed measurements
                if (movementSpeeds.length > 10) {
                    movementSpeeds.shift();
                }
                
                // Track direction changes
                const currentDirection = touchX > lastMousePos.x ? 'right' : touchX < lastMousePos.x ? 'left' : null;
                if (currentDirection && lastMovementDirection && currentDirection !== lastMovementDirection) {
                    movementDirectionChanges++;
                }
                lastMovementDirection = currentDirection;
                
                // Reset idle timer on movement
                if (deltaX > 1) { // Threshold to consider actual movement
                    // If there was an idle period, record it
                    const idleDuration = now - lastMovementTime;
                    if (idleDuration > 500) { // Idle threshold: 500ms
                        idlePeriods.push({
                            duration: idleDuration,
                            startTime: lastMovementTime,
                            endTime: now,
                            timeSinceLastDistraction: lastMovementTime - lastDistractionTime
                        });
                        
                        // Count as hesitation if close to a distraction
                        if (now - lastDistractionTime < 3000) { // Within 3 seconds of distraction
                            hesitations++;
                        }
                    }
                    lastMovementTime = now;
                }
            }
            
            // Record position for path analysis
            playerPositions.push({
                x: touchX,
                time: now,
                timeSinceDistraction: now - lastDistractionTime
            });
            
            // Limit the size of position history
            if (playerPositions.length > 100) {
                playerPositions.shift();
            }
            
            lastMousePos.x = touchX;
            lastMouseMoveTime = now;
            
            player.x = touchX;
            
            if (player.x < player.radius) {
                player.x = player.radius;
            } else if (player.x > canvas.width - player.radius) {
                player.x = canvas.width - player.radius;
            }
        }
    });
    
    canvas.addEventListener('touchend', function(e) {
        e.preventDefault();
        player.dragging = false;
    });
    
    // Start game event
    startButton.addEventListener('click', startGame);
    playAgainButton.addEventListener('click', function() {
        gameOverPanel.style.display = 'none';
        startGame();
    });
    
    // Draw the player
    function drawPlayer() {
        ctx.beginPath();
        ctx.arc(player.x, player.y, player.radius, 0, Math.PI * 2);
        ctx.fillStyle = player.color;
        ctx.fill();
        ctx.closePath();
    }
    
    // Create a new obstacle
    function createObstacle() {
        const width = 30;
        const height = 30;
        const x = Math.random() * (canvas.width - width);
        const creationTime = Date.now();
        
        totalObjects++; // Increment total objects
        
        obstacles.push({
            x: x,
            y: -height,
            width: width,
            height: height,
            color: 'red',
            creationTime: creationTime
        });
    }
    
    // Create a new reward
    function createReward() {
        const size = 20;
        const x = Math.random() * (canvas.width - size);
        const creationTime = Date.now();
        
        totalObjects++; // Increment total objects
        
        rewards.push({
            x: x,
            y: -size,
            size: size,
            color: 'green',
            creationTime: creationTime
        });
    }
    
    // Create a new emoji
    function createEmoji() {
        const size = 30;
        const x = Math.random() * (canvas.width - size);
        const emojiType = emojiTypes[Math.floor(Math.random() * emojiTypes.length)];
        const creationTime = Date.now();
        
        totalObjects++; // Increment total objects
        
        emojis.push({
            x: x,
            y: -size,
            size: size,
            type: emojiType.type,
            value: emojiType.value,
            emoji: emojiType.emoji,
            color: emojiType.color,
            creationTime: creationTime
        });
    }
    
    // Create a background flash
    function createFlash() {
        flashActive = true;
        lastDistractionTime = Date.now();
        distractionTimes.push(lastDistractionTime); // Record distraction time for analysis
        
        // Record pre-distraction speed
        preDistractionSpeed = getAverageMovementSpeed();
        
        // Schedule flash removal
        setTimeout(function() {
            flashActive = false;
            
            // Record post-distraction speed after a short delay
            setTimeout(function() {
                postDistractionSpeed = getAverageMovementSpeed();
                
                // Record distraction event data
                distractionEvents.push({
                    type: 'flash',
                    time: lastDistractionTime,
                    preSpeed: preDistractionSpeed,
                    postSpeed: postDistractionSpeed,
                    speedDelta: postDistractionSpeed - preDistractionSpeed
                });
            }, 1000);
        }, flashDuration);
    }
    
    // Calculate average movement speed
    function getAverageMovementSpeed() {
        if (movementSpeeds.length === 0) return 0;
        
        const sum = movementSpeeds.reduce((a, b) => a + b, 0);
        return sum / movementSpeeds.length;
    }
    
    // Draw obstacles
    function drawObstacles() {
        for (let i = 0; i < obstacles.length; i++) {
            const obstacle = obstacles[i];
            ctx.fillStyle = obstacle.color;
            ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
        }
    }
    
    // Draw rewards
    function drawRewards() {
        for (let i = 0; i < rewards.length; i++) {
            const reward = rewards[i];
            // Draw a star shape
            drawStar(reward.x + reward.size/2, reward.y + reward.size/2, 5, reward.size/2, reward.size/4);
        }
    }
    
    // Draw emojis
    function drawEmojis() {
        for (let i = 0; i < emojis.length; i++) {
            const emoji = emojis[i];
            
            // Draw background circle
            ctx.beginPath();
            ctx.arc(emoji.x + emoji.size/2, emoji.y + emoji.size/2, emoji.size/2, 0, Math.PI * 2);
            ctx.fillStyle = emoji.color;
            ctx.fill();
            ctx.closePath();
            
            // Draw emoji text
            ctx.font = emoji.size + 'px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(emoji.emoji, emoji.x + emoji.size/2, emoji.y + emoji.size/2);
        }
    }
    
    // Draw background flash
    function drawFlash() {
        if (flashActive) {
            ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
    }
    
    // Draw a star shape
    function drawStar(cx, cy, spikes, outerRadius, innerRadius) {
        let rot = Math.PI / 2 * 3;
        let x = cx;
        let y = cy;
        let step = Math.PI / spikes;
        
        ctx.beginPath();
        ctx.moveTo(cx, cy - outerRadius);
        
        for (let i = 0; i < spikes; i++) {
            x = cx + Math.cos(rot) * outerRadius;
            y = cy + Math.sin(rot) * outerRadius;
            ctx.lineTo(x, y);
            rot += step;
            
            x = cx + Math.cos(rot) * innerRadius;
            y = cy + Math.sin(rot) * innerRadius;
            ctx.lineTo(x, y);
            rot += step;
        }
        
        ctx.lineTo(cx, cy - outerRadius);
        ctx.closePath();
        ctx.fillStyle = 'green';
        ctx.fill();
    }
    
    // Update obstacle positions and check for collisions
    function updateObstacles() {
        for (let i = 0; i < obstacles.length; i++) {
            const obstacle = obstacles[i];
            obstacle.y += obstacleSpeed;
            
            // Check for collision with player
            if (
                player.x + player.radius > obstacle.x &&
                player.x - player.radius < obstacle.x + obstacle.width &&
                player.y + player.radius > obstacle.y &&
                player.y - player.radius < obstacle.y + obstacle.height
            ) {
                // Collision detected, reduce score
                score -= 1;
                blocksHit++; // Track blocks hit
                scoreDisplay.textContent = 'Score: ' + score;
                
                // Calculate reaction time
                const reactionTime = Date.now() - obstacle.creationTime;
                totalReactionTimes.push(reactionTime); // Add to all reaction times
                
                // Check if this is a post-distraction action
                if (isPostDistractionAction(obstacle.creationTime)) {
                    postDistractionReactionTimes.push(reactionTime);
                    postDistractionAttempts++;
                }
                
                reactionTimes.push({
                    type: 'obstacle_hit',
                    time: reactionTime,
                    timeSinceLastDistraction: Date.now() - lastDistractionTime
                });
                
                // Remove the obstacle
                obstacles.splice(i, 1);
                i--;
                obstacleAvoidances++;
            }
            
            // Remove obstacles that go off-screen
            else if (obstacle.y > canvas.height) {
                obstacles.splice(i, 1);
                i--;
                obstacleAvoidances++;
                blocksDodged++; // Track blocks dodged
                successfulActions++; // Count as successful dodge
            }
        }
        
        // Randomly create new obstacles
        if (Math.random() * obstacleFrequency < 1 && gameRunning) {
            createObstacle();
        }
    }
    
    // Update reward positions and check for collections
    function updateRewards() {
        for (let i = 0; i < rewards.length; i++) {
            const reward = rewards[i];
            reward.y += rewardSpeed;
            
            // Calculate distance between player center and reward center
            const dx = player.x - (reward.x + reward.size/2);
            const dy = player.y - (reward.y + reward.size/2);
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Check for collision with player
            if (distance < player.radius + reward.size/2) {
                // Collision detected, increase score
                score += 2; // Updated to +2 for stars
                starsCollected += 1;
                successfulActions++; // Count as successful collection
                scoreDisplay.textContent = 'Score: ' + score;
                
                // Calculate reaction time
                const reactionTime = Date.now() - reward.creationTime;
                totalReactionTimes.push(reactionTime); // Add to all reaction times
                
                // Check if this is a post-distraction action
                if (isPostDistractionAction(reward.creationTime)) {
                    postDistractionReactionTimes.push(reactionTime);
                    postDistractionSuccesses++;
                    postDistractionAttempts++;
                }
                
                reactionTimes.push({
                    type: 'reward_collected',
                    time: reactionTime,
                    timeSinceLastDistraction: Date.now() - lastDistractionTime
                });
                
                // Remove the reward
                rewards.splice(i, 1);
                i--;
            }
            
            // Remove rewards that go off-screen
            else if (reward.y > canvas.height) {
                rewards.splice(i, 1);
                i--;
                missedRewards++;
            }
        }
        
        // Randomly create new rewards
        if (Math.random() * rewardFrequency < 1 && gameRunning) {
            createReward();
        }
    }
    
    // Update emoji positions and check for collections
    function updateEmojis() {
        for (let i = 0; i < emojis.length; i++) {
            const emoji = emojis[i];
            emoji.y += emojiSpeed;
            
            // Calculate distance between player center and emoji center
            const dx = player.x - (emoji.x + emoji.size/2);
            const dy = player.y - (emoji.y + emoji.size/2);
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Check for collision with player
            if (distance < player.radius + emoji.size/2) {
                // Collision detected, adjust score based on emoji type
                // Update score calculation based on new requirements
                if (emoji.type === 'happy') {
                    score += 3; // Positive emojis now +3
                    positiveEmojisCollected++;
                    successfulActions++; // Count as successful collection
                } else if (emoji.type === 'sad') {
                    score -= 2; // Negative emojis now -2
                    negativeEmojisCollected++;
                    successfulActions++; // Count as successful collection
                } else {
                    score += emoji.value; // Keep original value for other types
                    successfulActions++; // Count as successful collection
                }
                
                scoreDisplay.textContent = 'Score: ' + score;
                
                // Calculate reaction time
                const reactionTime = Date.now() - emoji.creationTime;
                totalReactionTimes.push(reactionTime); // Add to all reaction times
                
                // Check if this is a post-distraction action
                if (isPostDistractionAction(emoji.creationTime)) {
                    postDistractionReactionTimes.push(reactionTime);
                    postDistractionSuccesses++;
                    postDistractionAttempts++;
                }
                
                // Record interaction with emoji
                const interactionData = {
                    type: 'emoji_' + emoji.type,
                    time: reactionTime,
                    value: emoji.value,
                    timeSinceLastDistraction: Date.now() - lastDistractionTime,
                    playerSpeed: getAverageMovementSpeed(),
                    playerPosition: player.x
                };
                
                reactionTimes.push(interactionData);
                
                // Enhanced emoji interaction tracking for Phase 3
                emotionalStimulusResponses.push({
                    type: emoji.type,
                    reactionTime: reactionTime,
                    playerPosition: player.x,
                    directionChangesBeforeCollection: movementDirectionChanges,
                    speedAtCollection: getAverageMovementSpeed(),
                    hesitationsBeforeCollection: hesitations,
                    timeSinceLastDistraction: Date.now() - lastDistractionTime
                });
                
                // Remove the emoji
                emojis.splice(i, 1);
                i--;
            }
            
            // Remove emojis that go off-screen
            else if (emoji.y > canvas.height) {
                emojis.splice(i, 1);
                i--;
            }
        }
        
        // Randomly create new emojis
        if (Math.random() * emojiFrequency < 1 && gameRunning) {
            createEmoji();
        }
    }
    
    // Helper function to check if an action occurred within 2 seconds after a distraction
    function isPostDistractionAction(objectCreationTime) {
        for (let i = 0; i < distractionTimes.length; i++) {
            const distractionTime = distractionTimes[i];
            const actionTime = Date.now();
            
            // If the object was created before the distraction and the action is within 2 seconds after
            if (objectCreationTime < distractionTime && actionTime - distractionTime <= 2000) {
                return true;
            }
        }
        return false;
    }
    
    // Main game loop
    function gameLoop() {
        if (!gameRunning) return;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw game objects
        drawFlash();  // Draw flash first so it appears behind other elements
        drawPlayer();
        drawObstacles();
        drawRewards();
        drawEmojis();
        
        // Update game objects
        updateObstacles();
        updateRewards();
        updateEmojis();
        
        // Continue the game loop
        requestAnimationFrame(gameLoop);
    }
    
    // Schedule periodic flashes
    function scheduleFlashes() {
        return setInterval(function() {
            if (gameRunning) {
                createFlash();
            }
        }, flashFrequency);
    }
    
    // Start the game
    function startGame() {
        // Reset game state
        gameRunning = true;
        gameTime = 60;
        score = 0;
        starsCollected = 0;
        obstacles = [];
        rewards = [];
        emojis = [];
        flashActive = false;
        reactionTimes = [];
        obstacleAvoidances = 0;
        missedRewards = 0;
        distractionEvents = [];
        movementSpeeds = [];
        lastDistractionTime = 0;
        
        // Reset critical data metrics
        totalReactionTimes = [];
        totalObjects = 0;
        successfulActions = 0;
        distractionTimes = [];
        postDistractionReactionTimes = [];
        postDistractionSuccesses = 0;
        postDistractionAttempts = 0;
        positiveEmojisCollected = 0;
        negativeEmojisCollected = 0;
        blocksHit = 0;
        blocksDodged = 0;
        
        // Reset new Phase 3 metrics
        movementDirectionChanges = 0;
        hesitations = 0;
        lastMovementDirection = null;
        playerPositions = [];
        idlePeriods = [];
        lastMovementTime = Date.now();
        emotionalStimulusResponses = [];
        
        player.x = canvas.width / 2;
        
        // Update display
        timerDisplay.textContent = 'Time: ' + gameTime + 's';
        scoreDisplay.textContent = 'Score: ' + score;
        startButton.style.display = 'none';
        
        // Start game loop
        gameLoop();
        
        // Start flash scheduler
        const flashTimer = scheduleFlashes();
        
        // Start timer
        timer = setInterval(function() {
            gameTime--;
            timerDisplay.textContent = 'Time: ' + gameTime + 's';
            
            if (gameTime <= 0) {
                endGame(flashTimer);
            }
        }, 1000);
    }
    
    // End the game
    function endGame(flashTimer) {
        gameRunning = false;
        clearInterval(timer);
        clearInterval(flashTimer);
        
        // Show game over panel
        gameOverPanel.style.display = 'block';
        finalScoreDisplay.textContent = 'Your score: ' + score;
        starsCollectedDisplay.textContent = 'Stars collected: ' + starsCollected;
        startButton.style.display = 'block';
        
        // Send game data to server
        sendGameData();
    }
    
    // Send game data to server for analysis
    function sendGameData() {
        // Calculate the critical metrics
        
        // 1. Average Reaction Time (ms)
        const avgReactionTime = totalReactionTimes.length > 0 ? 
            totalReactionTimes.reduce((sum, time) => sum + time, 0) / totalReactionTimes.length : 0;
        
        // 2. Accuracy (%)
        const accuracy = totalObjects > 0 ? 
            (successfulActions / totalObjects) * 100 : 0;
        
        // 3. Distraction Response
        // Calculate baseline metrics (excluding post-distraction periods)
        const baselineReactionTime = avgReactionTime;
        const baselineAccuracy = accuracy;
        
        // Calculate post-distraction metrics
        const postDistractionAvgReactionTime = postDistractionReactionTimes.length > 0 ?
            postDistractionReactionTimes.reduce((sum, time) => sum + time, 0) / postDistractionReactionTimes.length : 0;
        
        const postDistractionAccuracy = postDistractionAttempts > 0 ?
            (postDistractionSuccesses / postDistractionAttempts) * 100 : 0;
        
        // Calculate changes
        const reactionTimeChange = postDistractionAvgReactionTime - baselineReactionTime;
        const accuracyChange = postDistractionAccuracy - baselineAccuracy;
        
        // 4. Emoji Collection Ratio
        const emojiCollectionRatio = (positiveEmojisCollected + negativeEmojisCollected) > 0 ?
            positiveEmojisCollected / (positiveEmojisCollected + negativeEmojisCollected) : 0;
        
        // 5. Total Score (calculated throughout gameplay)
        // Final score is already tracked
        
        // Calculate metrics
        const averageRewardReactionTime = calculateAverageReactionTime('reward_collected');
        const averageObstacleReactionTime = calculateAverageReactionTime('obstacle_hit');
        const positiveEmojiInteractions = countEmojiInteractions('happy');
        const negativeEmojiInteractions = countEmojiInteractions('sad') + countEmojiInteractions('angry');
        const neutralEmojiInteractions = countEmojiInteractions('neutral');
        
        // Calculate average movement speed changes after distractions
        let avgSpeedDelta = 0;
        if (distractionEvents.length > 0) {
            const sum = distractionEvents.reduce((acc, event) => acc + event.speedDelta, 0);
            avgSpeedDelta = sum / distractionEvents.length;
        }
        
        // Calculate emotional bias
        let emotionalBias = 0;
        if (positiveEmojiInteractions + negativeEmojiInteractions > 0) {
            emotionalBias = (positiveEmojiInteractions - negativeEmojiInteractions) / 
                            (positiveEmojiInteractions + negativeEmojiInteractions);
        }
        
        // Phase 3: Calculate movement pattern metrics
        const avgHesitationDuration = calculateAverageHesitationDuration();
        const hesitationFrequency = idlePeriods.length / gameTime;
        const directionChangeFrequency = movementDirectionChanges / gameTime;
        const movementVariability = calculateMovementVariability();
        
        // Phase 3: Calculate emotional response metrics
        const avgResponseToPositive = calculateAvgResponseMetric('happy');
        const avgResponseToNegative = calculateAvgResponseMetric(['sad', 'angry']);
        const emotionalResponseRatio = avgResponseToPositive / (avgResponseToNegative || 1);
        
        const gameData = {
            // Critical data points
            avgReactionTime: avgReactionTime,
            accuracy: accuracy,
            distractionResponse: {
                reactionTimeChange: reactionTimeChange,
                accuracyChange: accuracyChange,
                baselineReactionTime: baselineReactionTime,
                postDistractionReactionTime: postDistractionAvgReactionTime,
                baselineAccuracy: baselineAccuracy,
                postDistractionAccuracy: postDistractionAccuracy
            },
            emojiCollectionRatio: emojiCollectionRatio,
            totalScore: score,
            
            // Additional metrics for detailed analysis
            blocksHit: blocksHit,
            blocksDodged: blocksDodged,
            positiveEmojisCollected: positiveEmojisCollected,
            negativeEmojisCollected: negativeEmojisCollected,
            totalObjects: totalObjects,
            successfulActions: successfulActions,
            
            // Original data points (keep these for backward compatibility)
            score: score,
            starsCollected: starsCollected,
            obstacleAvoidances: obstacleAvoidances,
            missedRewards: missedRewards,
            gameTime: 60, // Add game duration for calculations
            averageRewardReactionTime: averageRewardReactionTime,
            averageObstacleReactionTime: averageObstacleReactionTime,
            positiveEmojiInteractions: positiveEmojiInteractions,
            negativeEmojiInteractions: negativeEmojiInteractions,
            neutralEmojiInteractions: neutralEmojiInteractions,
            distractionResponseDelta: avgSpeedDelta,
            
            // New Phase 3 metrics
            movementDirectionChanges: movementDirectionChanges,
            hesitations: hesitations,
            movementVariability: movementVariability,
            avgHesitationDuration: avgHesitationDuration,
            hesitationFrequency: hesitationFrequency,
            directionChangeFrequency: directionChangeFrequency,
            avgResponseToPositive: avgResponseToPositive,
            avgResponseToNegative: avgResponseToNegative,
            emotionalResponseRatio: emotionalResponseRatio,
            
            // Detailed data
            reactionTimeDetail: reactionTimes,
            distractionEvents: distractionEvents,
            emotionalStimulusResponses: emotionalStimulusResponses,
            idlePeriods: idlePeriods,
            playerPositions: playerPositions.slice(-20) // Send only last 20 positions to reduce data size
        };
        
        console.log('Game data:', gameData);
        
        // Update game over screen with critical metrics
        document.getElementById('reactionTime').textContent = 
            `Average reaction time: ${Math.round(avgReactionTime)}ms`;
            
        // Add accuracy to the game over screen 
        let accuracyElem = document.getElementById('accuracy');
        if (!accuracyElem) {
            accuracyElem = document.createElement('li');
            accuracyElem.id = 'accuracy';
            document.querySelector('#resultDetails .metrics-section:nth-child(1) ul').appendChild(accuracyElem);
        }
        accuracyElem.textContent = `Accuracy: ${accuracy.toFixed(1)}%`;
        
        // Add distraction response to the game over screen
        let distractionElem = document.getElementById('distractionResponse');
        if (!distractionElem) {
            distractionElem = document.createElement('li');
            distractionElem.id = 'distractionResponse';
            document.querySelector('#resultDetails .metrics-section:nth-child(2) ul').appendChild(distractionElem);
        }
        distractionElem.textContent = `Distraction response: ${reactionTimeChange > 0 ? '+' : ''}${Math.round(reactionTimeChange)}ms reaction time, ${accuracyChange > 0 ? '+' : ''}${accuracyChange.toFixed(1)}% accuracy`;
        
        // Add emoji collection ratio to the game over screen
        let emojiRatioElem = document.getElementById('emojiRatio');
        if (!emojiRatioElem) {
            emojiRatioElem = document.createElement('li');
            emojiRatioElem.id = 'emojiRatio';
            document.querySelector('#resultDetails .metrics-section:nth-child(3) ul').appendChild(emojiRatioElem);
        }
        emojiRatioElem.textContent = `Emoji collection ratio: ${(emojiCollectionRatio * 100).toFixed(1)}% positive`;
        
        document.getElementById('emotionalBias').textContent = 
            `Emotional bias: ${emotionalBias.toFixed(2)} (${emotionalBias > 0 ? 'Positive' : emotionalBias < 0 ? 'Negative' : 'Neutral'})`;
        
        document.getElementById('distractionRecovery').textContent = 
            `Distraction recovery: ${avgSpeedDelta.toFixed(4)} (${avgSpeedDelta > 0 ? 'Faster' : avgSpeedDelta < 0 ? 'Slower' : 'No change'} after distractions)`;
            
        // Add Phase 3 metrics to the game over screen
        document.getElementById('hesitationMetric').textContent = 
            `Hesitation frequency: ${hesitationFrequency.toFixed(2)} per second`;
            
        document.getElementById('movementVariability').textContent = 
            `Movement variability: ${movementVariability.toFixed(2)}`;
            
        document.getElementById('emotionalResponseRatio').textContent = 
            `Emotional response ratio: ${emotionalResponseRatio.toFixed(2)} (${emotionalResponseRatio > 1 ? 'Prefers positive' : 'Prefers negative'})`;
        
        // Send data to the server
        fetch('/save_game_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(gameData),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            
            // Update emotional indicators section with data from server
            if (data.emotional_indicators && data.emotional_indicators.length > 0) {
                const indicatorsContainer = document.getElementById('emotionalIndicators');
                indicatorsContainer.innerHTML = ''; // Clear existing content
                
                data.emotional_indicators.forEach(indicator => {
                    const listItem = document.createElement('li');
                    listItem.innerHTML = `<strong>${indicator.emotion}</strong> (${Math.round(indicator.confidence * 100)}% confidence)<br>
                        <small>${indicator.indicators.join(', ')}</small>`;
                    indicatorsContainer.appendChild(listItem);
                });
            } else {
                document.getElementById('emotionalIndicators').innerHTML = 
                    '<li>No significant emotional indicators detected</li>';
            }
            
            // Add new features to the game over screen when available
            if (data.features) {
                const features = data.features;
                
                // Add Reaction Time Variability if available
                if ('reaction_time_variability' in features) {
                    const rtVariability = features.reaction_time_variability;
                    
                    // Create or update element for reaction time variability
                    let rtVariabilityElem = document.getElementById('reactionTimeVariability');
                    if (!rtVariabilityElem) {
                        rtVariabilityElem = document.createElement('li');
                        rtVariabilityElem.id = 'reactionTimeVariability';
                        document.querySelector('#resultDetails .metrics-section:nth-child(1) ul').appendChild(rtVariabilityElem);
                    }
                    
                    rtVariabilityElem.textContent = `Reaction Time Variability: ${Math.round(rtVariability)}ms`;
                }
                
                // Add Performance Degradation if available
                if ('performance_degradation' in features) {
                    const performanceDegradation = features.performance_degradation;
                    const degradationText = performanceDegradation > 0 
                        ? `Improvement: ${(performanceDegradation * 100).toFixed(1)}%` 
                        : `Degradation: ${(Math.abs(performanceDegradation) * 100).toFixed(1)}%`;
                    
                    let performanceElem = document.getElementById('performanceDegradation');
                    if (!performanceElem) {
                        performanceElem = document.createElement('li');
                        performanceElem.id = 'performanceDegradation';
                        document.querySelector('#resultDetails .metrics-section:nth-child(1) ul').appendChild(performanceElem);
                    }
                    
                    performanceElem.textContent = `Performance Over Time: ${degradationText}`;
                }
                
                // Add Emotional Avoidance Rate if available
                if ('emotional_stimuli_avoidance_rate' in features) {
                    const avoidanceRate = features.emotional_stimuli_avoidance_rate;
                    const avoidanceText = avoidanceRate > 0.6 
                        ? 'High avoidance of negative stimuli' 
                        : avoidanceRate > 0.4 
                            ? 'Moderate avoidance of negative stimuli' 
                            : 'Low avoidance of negative stimuli';
                    
                    let avoidanceElem = document.getElementById('emotionalAvoidance');
                    if (!avoidanceElem) {
                        avoidanceElem = document.createElement('li');
                        avoidanceElem.id = 'emotionalAvoidance';
                        document.querySelector('#resultDetails .metrics-section:nth-child(3) ul').appendChild(avoidanceElem);
                    }
                    
                    avoidanceElem.textContent = `Negative Stimuli Avoidance: ${(avoidanceRate * 100).toFixed(1)}% (${avoidanceText})`;
                }
                
                // Add Emoji Preferences if available
                if (features.emoji_preference_profile) {
                    const preferences = features.emoji_preference_profile;
                    
                    let preferencesElem = document.getElementById('emojiPreferences');
                    if (!preferencesElem) {
                        preferencesElem = document.createElement('li');
                        preferencesElem.id = 'emojiPreferences';
                        document.querySelector('#resultDetails .metrics-section:nth-child(3) ul').appendChild(preferencesElem);
                    }
                    
                    // Format the emoji preferences
                    const happyPct = (preferences.happy_preference * 100).toFixed(0);
                    const sadPct = (preferences.sad_preference * 100).toFixed(0);
                    const angryPct = (preferences.angry_preference * 100).toFixed(0);
                    const neutralPct = (preferences.neutral_preference * 100).toFixed(0);
                    
                    preferencesElem.innerHTML = `Emoji Preferences: 😊${happyPct}% 😢${sadPct}% 😠${angryPct}% 😐${neutralPct}%`;
                }
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            document.getElementById('emotionalIndicators').innerHTML = 
                '<li>Error analyzing emotional indicators</li>';
        });
    }
    
    // Helper function to calculate average reaction time by type
    function calculateAverageReactionTime(type) {
        const filteredTimes = reactionTimes.filter(item => item.type === type);
        if (filteredTimes.length === 0) return 0;
        
        const sum = filteredTimes.reduce((acc, item) => acc + item.time, 0);
        return sum / filteredTimes.length;
    }
    
    // Helper function to count emoji interactions by type
    function countEmojiInteractions(type) {
        return reactionTimes.filter(item => item.type === 'emoji_' + type).length;
    }
    
    // Phase 3: New helper functions for advanced metrics
    
    // Calculate average hesitation duration
    function calculateAverageHesitationDuration() {
        if (idlePeriods.length === 0) return 0;
        
        const sum = idlePeriods.reduce((acc, period) => acc + period.duration, 0);
        return sum / idlePeriods.length;
    }
    
    // Calculate movement variability (standard deviation of positions)
    function calculateMovementVariability() {
        if (playerPositions.length < 2) return 0;
        
        const positions = playerPositions.map(pos => pos.x);
        const mean = positions.reduce((a, b) => a + b, 0) / positions.length;
        
        const variance = positions.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / positions.length;
        return Math.sqrt(variance);
    }
    
    // Calculate average response metric for emotional stimulus types
    function calculateAvgResponseMetric(types) {
        if (!Array.isArray(types)) {
            types = [types];
        }
        
        const responses = emotionalStimulusResponses.filter(resp => types.includes(resp.type));
        
        if (responses.length === 0) return 0;
        
        // Normalize reaction time (lower is better)
        const avgReactionTime = responses.reduce((acc, r) => acc + r.reactionTime, 0) / responses.length;
        const normalizedTime = 1000 / (avgReactionTime || 1000); // Invert so faster = higher score
        
        return normalizedTime;
    }
    
    // Initial draw of the game
    drawPlayer();
}); 