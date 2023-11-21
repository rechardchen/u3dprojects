extends Node3D

@export var sensitivity := 5
var player

# Called when the node enters the scene tree for the first time.
func _ready():
	player = get_tree().get_nodes_in_group("player")[0]
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	global_position = player.global_position
	pass

func _input(event):
	if event is InputEventMouseMotion:
		var tempRot = rotation.x - event.relative.y / 1000 * sensitivity
		rotation.y -= event.relative.x / 1000 * sensitivity
		tempRot = clamp(tempRot, -1, 0.25)
		rotation.x = tempRot
