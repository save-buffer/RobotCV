# robot_cv
Computer vision for use on the Husky Robotics team.
One tracks a tennis ball, the other tracks a keyboard.

# Tennisball Tracker
Blurs the image, filters by color, then does basic algorithm to check if the object is circular.

# Keyboard Tracker
Divides the image up into keys, then uses neural network to identify keys. Will extrapolate locations of other keys if key division doesn't work correctly. 

![alt text](https://i.imgur.com/6IZELBC.png)

(Random picture my friend took of his keyboard)
