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
    let totalReactionTimes = [];
    let totalObjects = 0;
    let successfulActions = 0;
    let distractionTimes = [];
    let postDistractionReactionTimes = [];
    let postDistractionSuccesses = 0;
    let postDistractionAttempts = 0;
    let positiveEmojisCollected = 0;
    let negativeEmojisCollected = 0;
    let neutralEmojisCollected = 0;
    let blocksHit = 0;
    let blocksDodged = 0;
    let accuracyMetric = 0;
    let preDistractionAccuracy = 0;
    let postDistractionAccuracy = 0;
    let totalPreDistractionAttempts = 0;
    let totalPreDistractionSuccesses = 0;
    let emojiCollectionTotal = 0;
    let positiveEmojiPercentage = 0;
    let reactionTimeVariability = 0;
    let preGamePerformance = [];
    let postGamePerformance = [];
    let performanceDegradation = 0;
    
    // Metrics for data collection
    let obstacleAvoidances = 0;
    let missedRewards = 0;
    let lastDistractionTime = 0;
    let preDistractionSpeed = 0;
    let postDistractionSpeed = 0;
    let distractionEvents = [];
    let lastMouseMoveTime = 0;
    let lastMousePos = { x: 0 };
    let movementSpeeds = [];
    
    // Phase 3 metrics
    let movementDirectionChanges = 0;
    let hesitations = 0;
    let lastMovementDirection = null;
    let playerPositions = [];
    let idlePeriods = [];
    let lastMovementTime = Date.now();
    let emotionalStimulusResponses = [];
    let emotionalResponseRatio = 0;
    let movementVariability = 0;
    
    // Player object
    const player = {
        x: canvas.width / 2,
        y: canvas.height - 50,
        radius: 15,
        color: 'blue',
        speed: 5,
        dragging: false
    };
    
    // Game objects
    let obstacles = [];
    let rewards = [];
    let emojis = [];
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
    const obstacleFrequency = 60;
    const rewardFrequency = 120;
    const emojiFrequency = 150;
    const flashFrequency = 15000;
    const flashDuration = 500;
    
    // Object creation counters
    let frameCount = 0;
    
    // Mouse/Touch event handlers
    canvas.addEventListener('mousedown', startDragging);
    canvas.addEventListener('mousemove', handleDragging);
    canvas.addEventListener('mouseup', stopDragging);
    canvas.addEventListener('touchstart', handleTouchStart);
    canvas.addEventListener('touchmove', handleTouchMove);
    canvas.addEventListener('touchend', stopDragging);
    
    function startDragging(e) {
        const pos = getEventPosition(e);
        const dx = pos.x - player.x;
        const dy = pos.y - player.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < player.radius) {
            player.dragging = true;
            lastMousePos.x = pos.x;
            lastMouseMoveTime = Date.now();
        }
    }
    
    function handleDragging(e) {
        e.preventDefault(); // Prevent default behavior to avoid issues
        
        if (!gameRunning) return; // Don't process if game not running
        
        const pos = getEventPosition(e);
        if (player.dragging) {
            updatePlayerPosition(pos.x);
        } else {
            // Check if touch/click is directly on player to start dragging
            const dx = pos.x - player.x;
            const dy = pos.y - player.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < player.radius * 1.5) {
                player.dragging = true;
                lastMousePos.x = pos.x;
                lastMouseMoveTime = Date.now();
            }
        }
    }
    
    function stopDragging() {
        player.dragging = false;
    }
    
    function handleTouchStart(e) {
        e.preventDefault(); // Prevent scrolling
        if (e.touches && e.touches.length > 0) {
            const touch = e.touches[0];
            const pos = getEventPosition(touch);
            const dx = pos.x - player.x;
            const dy = pos.y - player.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < player.radius * 1.5) {
                player.dragging = true;
                lastMousePos.x = pos.x;
                lastMouseMoveTime = Date.now();
            }
        }
    }
    
    function handleTouchMove(e) {
        e.preventDefault(); // Prevent scrolling
        if (!gameRunning) return;
        
        if (player.dragging && e.touches && e.touches.length > 0) {
            const touch = e.touches[0];
            const pos = getEventPosition(touch);
            updatePlayerPosition(pos.x);
        }
    }
    
    function getEventPosition(e) {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX !== undefined) ? e.clientX : e.pageX;
        const y = (e.clientY !== undefined) ? e.clientY : e.pageY;
        return {
            x: x - rect.left,
            y: y - rect.top
        };
    }
    
    function updatePlayerPosition(x) {
        const now = Date.now();
        const deltaTime = now - lastMouseMoveTime;
        
        if (deltaTime > 0) {
            const deltaX = Math.abs(x - lastMousePos.x);
            const speed = deltaX / deltaTime;
            movementSpeeds.push(speed);
            
            if (movementSpeeds.length > 10) movementSpeeds.shift();
            
            const currentDirection = x > lastMousePos.x ? 'right' : x < lastMousePos.x ? 'left' : null;
            if (currentDirection && lastMovementDirection && currentDirection !== lastMovementDirection) {
                movementDirectionChanges++;
            }
            lastMovementDirection = currentDirection;
            
            if (deltaX > 1) {
                const idleDuration = now - lastMovementTime;
                if (idleDuration > 500) {
                    idlePeriods.push({
                        duration: idleDuration,
                        startTime: lastMovementTime,
                        endTime: now,
                        timeSinceLastDistraction: lastMovementTime - lastDistractionTime
                    });
                    
                    if (now - lastDistractionTime < 3000) {
                        hesitations++;
                    }
                }
                lastMovementTime = now;
            }
            
            playerPositions.push({
                x: x,
                time: now,
                timeSinceDistraction: now - lastDistractionTime
            });
            
            if (playerPositions.length > 100) playerPositions.shift();
            
            lastMousePos.x = x;
            lastMouseMoveTime = now;
            
            player.x = Math.max(player.radius, Math.min(x, canvas.width - player.radius));
        }
    }
    
    function drawPlayer() {
        ctx.beginPath();
        ctx.arc(player.x, player.y, player.radius, 0, Math.PI * 2);
        ctx.fillStyle = player.color;
        ctx.fill();
        ctx.closePath();
    }
    
    function createObstacle() {
        const obstacle = {
            x: Math.random() * (canvas.width - 30),
            y: -30,
            width: 30,
            height: 30,
            speed: obstacleSpeed,
            type: 'obstacle',
            creationTime: Date.now()
        };
        obstacles.push(obstacle);
        totalObjects++;
        
        // Track if this is post-distraction object
        const isPostDistraction = distractionTimes.some(time => Date.now() - time < 3000);
        if (isPostDistraction) {
            postDistractionAttempts++;
        } else {
            totalPreDistractionAttempts++;
        }
    }
    
    function createReward() {
        rewards.push({
            x: Math.random() * (canvas.width - 20),
            y: -20,
            radius: 10,
            speed: rewardSpeed,
            type: 'reward',
            creationTime: Date.now()
        });
        totalObjects++;
    }
    
    function createEmoji() {
        const emojiType = emojiTypes[Math.floor(Math.random() * emojiTypes.length)];
        emojis.push({
            x: Math.random() * (canvas.width - 30),
            y: -30,
            width: 30,
            height: 30,
            speed: emojiSpeed,
            type: emojiType.type,
            emoji: emojiType.emoji,
            value: emojiType.value,
            creationTime: Date.now()
        });
        totalObjects++;
    }
    
    function drawObstacles() {
        obstacles.forEach(obstacle => {
            ctx.fillStyle = 'red';
            ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
        });
    }
    
    function drawRewards() {
        rewards.forEach(reward => {
            drawStar(reward.x + reward.radius, reward.y + reward.radius, 5, reward.radius, reward.radius/2);
        });
    }
    
    function drawEmojis() {
        emojis.forEach(emoji => {
            ctx.font = '24px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(emoji.emoji, emoji.x + emoji.width/2, emoji.y + emoji.height/2);
            
            // Draw a light circle behind the emoji to make it more visible (optional)
            if (gameRunning) {
                ctx.beginPath();
                ctx.arc(emoji.x + emoji.width/2, emoji.y + emoji.height/2, emoji.width/2, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
                ctx.fill();
                ctx.closePath();
            }
        });
    }
    
    function drawStar(cx, cy, spikes, outerRadius, innerRadius) {
        let rot = Math.PI / 2 * 3;
        let x = cx;
        let y = cy;
        let step = Math.PI / spikes;

        ctx.beginPath();
        ctx.moveTo(cx, cy - outerRadius);
        
        for(let i = 0; i < spikes; i++) {
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
    
    function checkCollision(player, object) {
        // For circular player and rectangular or circular objects
        let objectCenterX, objectCenterY, objectWidth, objectHeight;
        
        if (object.radius) {
            // For circular objects (rewards)
            objectCenterX = object.x + object.radius;
            objectCenterY = object.y + object.radius;
            objectWidth = object.radius * 2;
            objectHeight = object.radius * 2;
        } else {
            // For rectangular objects (obstacles and emojis)
            objectCenterX = object.x + object.width / 2;
            objectCenterY = object.y + object.height / 2;
            objectWidth = object.width;
            objectHeight = object.height;
        }
        
        // Calculate closest point on rectangle to circle
        const closestX = Math.max(object.x, Math.min(player.x, object.x + objectWidth));
        const closestY = Math.max(object.y, Math.min(player.y, object.y + objectHeight));
        
        // Calculate distance between closest point and circle center
        const distanceX = player.x - closestX;
        const distanceY = player.y - closestY;
        const distanceSquared = distanceX * distanceX + distanceY * distanceY;
        
        // Simplified circular collision with slightly increased radius for better gameplay experience
        return distanceSquared < (player.radius * player.radius * 1.2);
    }
    
    function updateGameObjects() {
        // Update obstacles
        for (let i = 0; i < obstacles.length; i++) {
            obstacles[i].y += obstacles[i].speed;
            
            // Check for collision with player
            if (checkCollision(player, obstacles[i])) {
                score -= 1;
                scoreDisplay.textContent = `Score: ${score}`;
                
                // Record interaction time for reaction time calculation
                const reactionTime = Date.now() - obstacles[i].creationTime;
                totalReactionTimes.push(reactionTime);
                
                // Check if this happened after a distraction
                const isPostDistraction = distractionTimes.some(time => Date.now() - time < 3000);
                if (isPostDistraction) {
                    postDistractionReactionTimes.push(reactionTime);
                }
                
                blocksHit++;
                obstacles.splice(i, 1);
                i--;
                continue;
            }
            
            // Check if obstacle passed the player without collision
            if (obstacles[i].y > canvas.height) {
                blocksDodged++;
                successfulActions++;
                
                // Check if this happened after a distraction
                const isPostDistraction = distractionTimes.some(time => Date.now() - time < 3000);
                if (isPostDistraction) {
                    postDistractionSuccesses++;
                } else {
                    totalPreDistractionSuccesses++;
                }
                
                // Track early vs late game performance
                if (gameTime > 30) {
                    preGamePerformance.push(1); // Success in first half
                } else {
                    postGamePerformance.push(1); // Success in second half
                }
                
                obstacles.splice(i, 1);
                i--;
            }
        }
        
        // Update rewards
        for (let i = 0; i < rewards.length; i++) {
            rewards[i].y += rewards[i].speed;
            
            // Check for collision with player
            if (checkCollision(player, rewards[i])) {
                score += 1;
                starsCollected++;
                scoreDisplay.textContent = `Score: ${score}`;
                
                // Record interaction time for reaction time calculation
                const reactionTime = Date.now() - rewards[i].creationTime;
                totalReactionTimes.push(reactionTime);
                
                // Check if this happened after a distraction
                const isPostDistraction = distractionTimes.some(time => Date.now() - time < 3000);
                if (isPostDistraction) {
                    postDistractionReactionTimes.push(reactionTime);
                    postDistractionSuccesses++;
                } else {
                    totalPreDistractionSuccesses++;
                }
                
                // Track early vs late game performance
                if (gameTime > 30) {
                    preGamePerformance.push(1); // Success in first half
                } else {
                    postGamePerformance.push(1); // Success in second half
                }
                
                successfulActions++;
                rewards.splice(i, 1);
                i--;
                continue;
            }
            
            // Check if reward passed the player without being collected
            if (rewards[i].y > canvas.height) {
                missedRewards++;
                
                // Track early vs late game performance
                if (gameTime > 30) {
                    preGamePerformance.push(-1); // Failure in first half
                } else {
                    postGamePerformance.push(-1); // Failure in second half
                }
                
                rewards.splice(i, 1);
                i--;
            }
        }
        
        // Update emojis
        for (let i = 0; i < emojis.length; i++) {
            emojis[i].y += emojis[i].speed;
            
            // Check for collision with player
            if (checkCollision(player, emojis[i])) {
                score += emojis[i].value;
                scoreDisplay.textContent = `Score: ${score}`;
                
                // Track emoji type collected
                emojiCollectionTotal++;
                if (emojis[i].type === 'happy') {
                    positiveEmojisCollected++;
                } else if (emojis[i].type === 'sad' || emojis[i].type === 'angry') {
                    negativeEmojisCollected++;
                } else if (emojis[i].type === 'neutral') {
                    neutralEmojisCollected++;
                }
                
                // Record interaction for emotional stimulus response
                const reactionTime = Date.now() - emojis[i].creationTime;
                emotionalStimulusResponses.push({
                    type: emojis[i].type,
                    reactionTime: reactionTime,
                    value: emojis[i].value,
                    timeSinceDistraction: Date.now() - lastDistractionTime
                });
                
                // Add to total reaction times
                totalReactionTimes.push(reactionTime);
                
                // Check if this happened after a distraction
                const isPostDistraction = distractionTimes.some(time => Date.now() - time < 3000);
                if (isPostDistraction) {
                    postDistractionReactionTimes.push(reactionTime);
                    postDistractionSuccesses++;
                } else {
                    totalPreDistractionSuccesses++;
                }
                
                successfulActions++;
                emojis.splice(i, 1);
                i--;
                
                // Track early vs late game performance
                if (gameTime > 30) {
                    preGamePerformance.push(1); // Success in first half
                } else {
                    postGamePerformance.push(1); // Success in second half
                }
                
                continue;
            }
            
            // Check if emoji passed the player without being collected
            if (emojis[i].y > canvas.height) {
                emojis.splice(i, 1);
                i--;
                
                // Track early vs late game performance
                if (gameTime > 30) {
                    preGamePerformance.push(-1); // Failure in first half
                } else {
                    postGamePerformance.push(-1); // Failure in second half
                }
            }
        }
    }
    
    function gameLoop() {
        if (!gameRunning) return;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Increment frame counter
        frameCount++;
        
        // Create game objects based on frame count
        if (frameCount % obstacleFrequency === 0) createObstacle();
        if (frameCount % rewardFrequency === 0) createReward();
        if (frameCount % emojiFrequency === 0) createEmoji();
        
        updateGameObjects();
        
        drawPlayer();
        drawObstacles();
        drawRewards();
        drawEmojis();
        
        if (flashActive) {
            ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
        
        requestAnimationFrame(gameLoop);
    }
    
    function calculateMovementVariability() {
        if (playerPositions.length < 2) return 0;
        
        // Calculate standard deviation of positions
        const positions = playerPositions.map(p => p.x);
        const mean = positions.reduce((a, b) => a + b, 0) / positions.length;
        const squareDiffs = positions.map(p => (p - mean) ** 2);
        const variance = squareDiffs.reduce((a, b) => a + b, 0) / squareDiffs.length;
        
        return Math.sqrt(variance);
    }
    
    function calculateEmotionalResponseRatio() {
        // Return 0 if no emojis collected
        if (positiveEmojisCollected + negativeEmojisCollected === 0) {
            console.log("No positive or negative emojis collected for response ratio");
            return 0;
        }
        
        const ratio = (positiveEmojisCollected - negativeEmojisCollected) / 
                (positiveEmojisCollected + negativeEmojisCollected);
        console.log(`Calculated emotional response ratio: ${ratio.toFixed(2)} from ${positiveEmojisCollected} positive and ${negativeEmojisCollected} negative emojis`);
        return ratio;
    }
    
    function calculateAverageReactionTime() {
        if (totalReactionTimes.length === 0) return 0;
        return totalReactionTimes.reduce((a, b) => a + b, 0) / totalReactionTimes.length;
    }
    
    function calculateAccuracy() {
        if (totalObjects === 0) return 0;
        return (successfulActions / totalObjects) * 100;
    }
    
    function calculateReactionTimeVariability() {
        if (totalReactionTimes.length <= 1) return 0;
        
        // Calculate standard deviation of reaction times
        const mean = totalReactionTimes.reduce((a, b) => a + b, 0) / totalReactionTimes.length;
        const squareDiffs = totalReactionTimes.map(time => (time - mean) ** 2);
        const variance = squareDiffs.reduce((a, b) => a + b, 0) / squareDiffs.length;
        
        return Math.sqrt(variance);
    }
    
    function calculatePerformanceDegradation() {
        if (preGamePerformance.length === 0 || postGamePerformance.length === 0) return 0;
        
        const preAvg = preGamePerformance.reduce((a, b) => a + b, 0) / preGamePerformance.length;
        const postAvg = postGamePerformance.reduce((a, b) => a + b, 0) / postGamePerformance.length;
        
        return (postAvg - preAvg) / Math.abs(preAvg) * 100;
    }
    
    function calculatePositiveEmojiPercentage() {
        if (emojiCollectionTotal === 0) return 0;
        return (positiveEmojisCollected / emojiCollectionTotal) * 100;
    }
    
    function calculateDistractionResponseDelta() {
        // Calculate reaction time change after distractions
        const preDistractionAvgTime = totalReactionTimes.filter(time => 
            !distractionTimes.some(distTime => 
                time > distTime && time < distTime + 3000
            )
        ).reduce((a, b) => a + b, 0) / Math.max(1, totalReactionTimes.filter(time => 
            !distractionTimes.some(distTime => 
                time > distTime && time < distTime + 3000
            )
        ).length);
        
        const postDistractionAvgTime = postDistractionReactionTimes.reduce((a, b) => a + b, 0) / 
                                      Math.max(1, postDistractionReactionTimes.length);
        
        return postDistractionAvgTime - preDistractionAvgTime;
    }
    
    function calculateDistractionAccuracyDelta() {
        if (totalPreDistractionAttempts === 0 || postDistractionAttempts === 0) return 0;
        
        preDistractionAccuracy = (totalPreDistractionSuccesses / totalPreDistractionAttempts) * 100;
        postDistractionAccuracy = (postDistractionSuccesses / postDistractionAttempts) * 100;
        
        return postDistractionAccuracy - preDistractionAccuracy;
    }
    
    function startGame() {
        console.log("startGame function called");
        if (gameRunning) return;
        
        // Reset game state
        gameRunning = true;
        score = 0;
        starsCollected = 0;
        obstacles = [];
        rewards = [];
        emojis = [];
        gameTime = 60;
        frameCount = 0;
        
        // Reset metrics
        totalReactionTimes = [];
        totalObjects = 0;
        successfulActions = 0;
        distractionTimes = [];
        postDistractionReactionTimes = [];
        postDistractionSuccesses = 0;
        postDistractionAttempts = 0;
        positiveEmojisCollected = 0;
        negativeEmojisCollected = 0;
        neutralEmojisCollected = 0;
        blocksHit = 0;
        blocksDodged = 0;
        movementDirectionChanges = 0;
        hesitations = 0;
        playerPositions = [];
        idlePeriods = [];
        emotionalStimulusResponses = [];
        preGamePerformance = [];
        postGamePerformance = [];
        
        // Update display
        scoreDisplay.textContent = `Score: ${score}`;
        timerDisplay.textContent = `Time: ${gameTime}s`;
        gameOverPanel.style.display = 'none';
        startButton.style.display = 'none';
        
        // Ensure player is at starting position
        player.x = canvas.width / 2;
        player.y = canvas.height - 50;
        
        // Start game loop
        gameLoop();
        
        // Start timer
        timer = setInterval(() => {
            gameTime--;
            timerDisplay.textContent = `Time: ${gameTime}s`;
            
            if (gameTime <= 0) {
                endGame();
            }
        }, 1000);
        
        // Schedule flashes
        const flashTimer = setInterval(() => {
            if (!gameRunning) {
                clearInterval(flashTimer);
                return;
            }
            
            flashActive = true;
            lastDistractionTime = Date.now();
            distractionTimes.push(lastDistractionTime);
            
            // Record pre-distraction speed (average of last few movements)
            if (movementSpeeds.length > 0) {
                preDistractionSpeed = movementSpeeds.reduce((a, b) => a + b, 0) / movementSpeeds.length;
            }
            
            setTimeout(() => {
                flashActive = false;
                
                // Record post-distraction speed after flash ends
                if (movementSpeeds.length > 0) {
                    postDistractionSpeed = movementSpeeds.reduce((a, b) => a + b, 0) / movementSpeeds.length;
                }
            }, flashDuration);
        }, flashFrequency);
    }
    
    function updateMetricsDisplay(data) {
        console.log("Updating metrics display");
        try {
            // Calculate all metrics
            const avgReactionTime = calculateAverageReactionTime() || 0;
            accuracyMetric = calculateAccuracy() || 0;
            emotionalResponseRatio = calculateEmotionalResponseRatio() || 0;
            const distractionRecovery = postDistractionSpeed > 0 && preDistractionSpeed > 0 ? 
                (postDistractionSpeed / preDistractionSpeed).toFixed(4) : '0.0000';
            const hesitationFreq = gameTime > 0 ? (hesitations / 60).toFixed(2) : '0.00';
            movementVariability = calculateMovementVariability().toFixed(2) || '0.00';
            
            // Calculate emotional bias (safely)
            let emotionalBias = 0;
            if (positiveEmojisCollected + negativeEmojisCollected > 0) {
                emotionalBias = (positiveEmojisCollected - negativeEmojisCollected) / 
                              Math.max(1, positiveEmojisCollected + negativeEmojisCollected);
            }
            
            const distractionResponseDelta = calculateDistractionResponseDelta().toFixed(0) || '0';
            const distractionAccuracyDelta = calculateDistractionAccuracyDelta().toFixed(1) || '0.0';
            reactionTimeVariability = calculateReactionTimeVariability().toFixed(2) || '0.00';
            performanceDegradation = calculatePerformanceDegradation().toFixed(2) || '0.00';
            positiveEmojiPercentage = calculatePositiveEmojiPercentage().toFixed(1) || '0.0';
            
            console.log("Calculated metrics:", {
                avgReactionTime,
                accuracyMetric,
                emotionalResponseRatio,
                distractionRecovery,
                movementVariability,
                emotionalBias,
                reactionTimeVariability,
                positiveEmojiPercentage
            });
            
            // Safely update element content
            const updateElement = (id, text) => {
                const element = document.getElementById(id);
                if (element) {
                    element.textContent = text;
                } else {
                    console.error(`Element with id ${id} not found`);
                }
            };
            
            // Update the display with safe element updates
            updateElement('reactionTime', `Average reaction time: ${Math.round(avgReactionTime)}ms`);
            updateElement('accuracyMetric', `Accuracy: ${accuracyMetric.toFixed(1)}%`);
            updateElement('emotionalResponseRatio', `Emotional response ratio: ${emotionalResponseRatio.toFixed(2)}`);
            updateElement('distractionRecovery', `Distraction recovery: ${distractionRecovery}`);
            updateElement('hesitationMetric', `Hesitation frequency: ${hesitationFreq} per second`);
            updateElement('movementVariability', `Movement variability: ${movementVariability}`);
            updateElement('emotionalBias', `Emotional bias: ${emotionalBias.toFixed(2)}`);
            updateElement('distractionResponseDelta', `Distraction response: ${distractionResponseDelta}ms reaction time, ${distractionAccuracyDelta}% accuracy`);
            updateElement('reactionTimeVariability', `Reaction time variability: ${reactionTimeVariability}`);
            updateElement('performanceDegradation', `Performance change: ${performanceDegradation}%`);
            updateElement('emojiCollectionPercentage', `Emoji collection ratio: ${positiveEmojiPercentage}% positive`);
            
            // Log that metrics display was updated successfully
            console.log("Metrics display updated successfully");
        } catch (error) {
            console.error("Error updating metrics display:", error);
        }
    }
    
    function endGame() {
        console.log("Game over - calculating final metrics");
        gameRunning = false;
        clearInterval(timer);
        
        try {
            // Calculate final metrics
            accuracyMetric = calculateAccuracy();
            reactionTimeVariability = calculateReactionTimeVariability();
            performanceDegradation = calculatePerformanceDegradation();
            positiveEmojiPercentage = calculatePositiveEmojiPercentage();
            
            console.log("Final metrics calculated:", {
                accuracy: accuracyMetric,
                reactionTimeVar: reactionTimeVariability,
                performanceDeg: performanceDegradation,
                emojiPercent: positiveEmojiPercentage
            });
            
            // Update metrics display immediately
            updateMetricsDisplay();
            
            // Update basic result display immediately
            finalScoreDisplay.textContent = `Final Score: ${score}`;
            starsCollectedDisplay.textContent = `Stars Collected: ${starsCollected}`;
            
            // Show game over panel right away
            gameOverPanel.style.display = 'block';
            startButton.style.display = 'block';
            
            // Calculate final metrics for server
            const gameData = {
                score: score,
                starsCollected: starsCollected,
                obstacleAvoidances: obstacleAvoidances,
                missedRewards: missedRewards,
                totalObjects: totalObjects,
                successfulActions: successfulActions,
                blocksHit: blocksHit,
                blocksDodged: blocksDodged,
                accuracy: accuracyMetric,
                positiveEmojiInteractions: positiveEmojisCollected,
                negativeEmojiInteractions: negativeEmojisCollected,
                neutralEmojiInteractions: neutralEmojisCollected,
                movementDirectionChanges: movementDirectionChanges,
                hesitations: hesitations,
                distractionResponseDelta: postDistractionSpeed - preDistractionSpeed,
                distractionAccuracyDelta: calculateDistractionAccuracyDelta(),
                movementVariability: movementVariability,
                emotionalResponseRatio: emotionalResponseRatio,
                reactionTimes: totalReactionTimes,
                reactionTimeVariability: reactionTimeVariability,
                performanceDegradation: performanceDegradation,
                positiveEmojiPercentage: positiveEmojiPercentage,
                avgResponseToPositive: emotionalStimulusResponses
                    .filter(r => r.type === 'happy')
                    .reduce((sum, r) => sum + r.reactionTime, 0) / 
                    Math.max(1, emotionalStimulusResponses.filter(r => r.type === 'happy').length),
                avgResponseToNegative: emotionalStimulusResponses
                    .filter(r => ['sad', 'angry'].includes(r.type))
                    .reduce((sum, r) => sum + r.reactionTime, 0) / 
                    Math.max(1, emotionalStimulusResponses.filter(r => ['sad', 'angry'].includes(r.type)).length),
                preDistractionSpeed: preDistractionSpeed,
                postDistractionSpeed: postDistractionSpeed,
                preDistractionAccuracy: preDistractionAccuracy,
                postDistractionAccuracy: postDistractionAccuracy,
                emotionalStimulusResponses: emotionalStimulusResponses,
                playerPositions: playerPositions,
                idlePeriods: idlePeriods,
                distractionEvents: distractionTimes
            };
            
            // Send data to server
            fetch('/save_game_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(gameData)
            })
            .then(response => {
                console.log("Server response received");
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log("Processing server data:", data);
                
                // Display emotional indicators
                const indicatorsList = document.getElementById('emotionalIndicators');
                if (!indicatorsList) {
                    console.error("Emotional indicators list element not found");
                    return;
                }
                
                indicatorsList.innerHTML = '';
                
                if (data.emotional_indicators && data.emotional_indicators.length > 0) {
                    data.emotional_indicators.forEach(indicator => {
                        const li = document.createElement('li');
                        li.textContent = `${indicator.emotion} (${Math.round(indicator.confidence * 100)}% confidence)`;
                        indicatorsList.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.textContent = 'No significant emotional indicators detected';
                    indicatorsList.appendChild(li);
                }
            })
            .catch(error => {
                console.error('Error sending/receiving game data:', error);
                
                // Still show something in the emotional indicators section
                const indicatorsList = document.getElementById('emotionalIndicators');
                if (indicatorsList) {
                    indicatorsList.innerHTML = '<li>Unable to analyze emotional indicators - connection issue</li>';
                }
            });
        } catch (error) {
            console.error("Error in endGame function:", error);
            
            // Ensure game over panel is displayed even if there's an error
            if (gameOverPanel) {
                gameOverPanel.style.display = 'block';
            }
            if (startButton) {
                startButton.style.display = 'block';
            }
        }
    }
    
    // Event listeners for buttons
    startButton.addEventListener('click', function(event) {
        console.log('Start button clicked');
        event.preventDefault();
        startGame();
    });

    playAgainButton.addEventListener('click', function(event) {
        console.log('Play Again button clicked');
        event.preventDefault();
        startGame();
    });

    // Log to console to verify script load
    console.log('Game script loaded successfully. Start button ready.');
}); 