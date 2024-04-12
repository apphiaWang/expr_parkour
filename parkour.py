import os
import sys
import cv2
import random
import pygame
import pygame.camera
from deepface import DeepFace
from au_model import create_model, get_expression_confidence_scores
import numpy as np

# init game setting
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 380

# size for parkour lane
LANE_WIDTH = 800
LANE_HEIGHT = 200

# size for webcam window
CAM_WIDTH = 240
CAM_HEIGHT = 180

GAME_SPEED = 5

DELTA_TO_GROUND = 18

ENERGY_DASH_DURATION = 120

ENERGY_GROWTH_DELTA = 100

# init camera through opencv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# initialize au detector
au_detector = create_model()

# init pygame
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode( (SCREEN_WIDTH, SCREEN_HEIGHT) )

pygame.display.set_caption('Expression Parkour Master')

class BG:
	def __init__(self, x): 
		self.width = LANE_WIDTH
		self.height = LANE_HEIGHT
		self.x = x
		self.y = 0
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx
		if self.x <= -LANE_WIDTH:
			self.x = LANE_WIDTH

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join('assets/images/bg_new.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class Instruction:
	def __init__(self): 
		self.width = SCREEN_WIDTH - CAM_WIDTH
		self.height = CAM_HEIGHT
		self.x = CAM_WIDTH
		self.y = LANE_HEIGHT
		self.font = pygame.font.SysFont('monospace', 14)
		self.color = (0, 0, 0)
		self.show()

	def show(self):
		screen.fill((175, 200, 173), rect = (self.x, self.y, self.width, self.height))
		instr = """ How To Play:
		- Make a SMILEY face to jump over obstacle like Rock and Bush
		- Make a SURPRISE face to do a slide shovel below a ghost fire
		- When the enegery bar is full, make an ANGRY face to dash
		Rules:
		- You will die if you hit a rock or a ghost fire
		- You won't die if you hit a bush, but your energy will be cleared,
		and the number of enemy chasing you will increase by 1
		- When there are 3 enemies chasing you, you will be caught
		- You are unrivaled during dash
		"""
		for idx, line in enumerate(instr.split('\n')):
			lbl = self.font.render(line.strip(), 1, self.color)
			screen.blit(lbl, (self.x + 10, self.y + 5 + 17 * idx))

class Enemy:
	def __init__(self):
		self.width = 55
		self.height = 50
		self.x = 0
		self.y = LANE_HEIGHT - self.height - 30
		self.texture_num = 0
		self.number = 1
		self.set_texture()
		self.show()

	def update(self, loops):
		self.texture_num = int(loops/10) % 3
		self.set_texture()

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def add_monster(self):
		self.number += 1
		self.x += 15
		self.set_texture()
		self.show()

	def reduce_monster(self):
		if self.number > 1:
			self.number -= 1
			self.x -= 10
			self.set_texture()
	
	def reset_monster(self):
		self.number = 1
		self.x = 10
		self.set_texture()
	
	def set_texture(self):
		path = os.path.join(f'assets/images/ghost{self.number}_{self.texture_num}.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class MC:
	def __init__(self):
		self.width = 48
		self.height = 60
		self.x = 80
		self.y = LANE_HEIGHT - 60 - DELTA_TO_GROUND
		self.texture_img = 'idle.png'
		self.texture_num = 0
		self.dy = 2
		self.gravity = 1.2
		self.onground = True
		self.jumping = False
		self.jump_stop = 40
		self.falling = False
		self.fall_stop = self.y
		self.shoveling = False
		self.shovel_stop = 0
		self.dashing = False
		self.dash_stop = 0
		self.set_texture()
		self.set_sound()
		self.show()

	def update(self, loops):
		# jumping
		if self.jumping:
			self.y -= self.dy
			if self.y <= self.jump_stop:
				self.fall()
		
		# falling after jumping
		elif self.falling:
			self.y += self.gravity * self.dy
			if self.y >= self.fall_stop:
				self.stop()

		# shoveling
		elif self.shoveling:
			if loops > self.shovel_stop:
				self.shoveling = False
				self.reset_size()
			else:
				self.texture_num = 0
				self.texture_img = 'shovel.png'
				self.set_texture()
		
		# stop dashing on time
		elif self.dashing:
			if loops > self.dash_stop:
				self.dashing = False
				self.reset_size()
			else:
				self.texture_num = (self.texture_num + 1) % 6
				self.texture_img = f'dash{self.texture_num}.png'
				self.set_texture()
		# running
		elif self.onground and loops % 4 == 0:
			self.texture_num = (self.texture_num + 1) % 3
			self.texture_img = f'run{self.texture_num}.png'
			self.set_texture()

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join(f'assets/images/{self.texture_img}')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

	def set_sound(self):
		path = os.path.join('assets/sounds/jump.wav')
		self.sound = pygame.mixer.Sound(path)

	def shovel(self, loops):
		# self.sound.play()
		self.shoveling = True
		self.shovel_stop = loops + 60
		self.shovel_resize()

	def dash_resize(self):
		self.width = 90
		self.height = 60
		self.y = LANE_HEIGHT - 60 - DELTA_TO_GROUND

	def shovel_resize(self):
		self.width = 60
		self.height = 40
		self.y = LANE_HEIGHT - 40 - DELTA_TO_GROUND

	def reset_size(self):
		self.width = 48
		self.height = 60
		self.y = LANE_HEIGHT - 60 - DELTA_TO_GROUND

	def dash(self, loops):
		# self.sound.play()
		self.dashing = True
		self.dash_stop = loops + ENERGY_DASH_DURATION
		self.dash_resize()

	def jump(self):
		self.sound.play()
		self.jumping = True
		self.onground = False
		self.texture_img = 'jump.png'
		self.set_texture()

	def fall(self):
		self.jumping = False
		self.falling = True

	def in_action(self):
		return self.jumping or self.falling or self.shoveling or self.dashing
	
	def stop(self):
		self.falling = False
		self.onground = True

class Obstacle:
	def __init__(self):
		self.hit = False
		self.crash = False
		self.set_sound()
	
	def get_hit(self):
		self.hit = True
	
	def get_crash(self):
		self.crash = True
		self.set_crash_texture()
		self.crash_sound.play()
	
	def set_crash_texture(self):
		path = os.path.join('assets/images/crash.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (30, 30))
	
	def set_sound(self):
		path = os.path.join('assets/sounds/crash.mp3')
		self.crash_sound = pygame.mixer.Sound(path)
	
	def check_collision(self, character):
		DELTA_H = -22
		DELTA_V = -12
		if self.x - character.x - character.width >= DELTA_H:
			return False
		if character.x - self.x - self.width >= DELTA_H:
			return False
		if self.y - character.y - character.height >= DELTA_V:
			return False
		if character.y - self.y - self.height >= DELTA_V:
			return False
		return True
		
class Bush(Obstacle):
	def __init__(self, x):
		super().__init__()
		self.width = 25
		self.height = 25
		self.x = x
		self.y = LANE_HEIGHT - 25 - DELTA_TO_GROUND
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join('assets/images/bush.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class Rock(Obstacle):	
	def __init__(self, x):
		super().__init__()
		self.width = 30
		self.height = 45
		self.x = x
		self.y = LANE_HEIGHT - 45 - DELTA_TO_GROUND
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx	

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join('assets/images/rock.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class Fire(Obstacle):
	def __init__(self, x):
		super().__init__()
		self.width = 55
		self.height = 75
		self.x = x
		self.y = LANE_HEIGHT - 75 - 55
		self.set_texture()
		self.show()

	def update(self, dx):
		self.x += dx	

	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join('assets/images/fire.png')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

class EnergyBar:
	def __init__(self):
		self.value = 0
		self.growth = 0
		self.width = 100
		self.height = 30
		self.x = LANE_WIDTH - 120
		self.y = 10
		self.energy_dash_stop = -1
		self.texture_img = 'energy_0.png'
		self.set_texture()
		self.set_sound()
		self.show()

	def update(self, loops):
		if (self.value < 5):
			self.growth += 1
			if self.growth == ENERGY_GROWTH_DELTA:
				self.value += 1
				if self.value == 5:
					self.full_sound.play()
				else:
					self.point_sound.play()
				self.growth = 0
				self.texture_img = f'energy_{self.value}.png'
				self.set_texture()
		if (self.value == 5 and loops == self.energy_dash_stop):
			self.value = 0
			self.texture_img = f'energy_{self.value}.png'
			self.set_texture()

	def clear_energy(self):
		self.value = 0
		self.texture_img = f'energy_{self.value}.png'
		self.clear_sound.play()
		self.set_texture()
	
	def use_energy(self, loops):
		self.energy_dash_stop = loops + ENERGY_DASH_DURATION
		self.use_sound.play()
		self.texture_img = f'energy_using.png'
		self.set_texture()
	
	def show(self):
		screen.blit(self.texture, (self.x, self.y))

	def set_texture(self):
		path = os.path.join(f'assets/images/{self.texture_img}')
		self.texture = pygame.image.load(path)
		self.texture = pygame.transform.scale(self.texture, (self.width, self.height))

	def set_sound(self):
		path = os.path.join('assets/sounds/point.wav')
		self.point_sound = pygame.mixer.Sound(path)
		path = os.path.join('assets/sounds/full.mp3')
		self.full_sound = pygame.mixer.Sound(path)
		path = os.path.join('assets/sounds/clear.mp3')
		self.clear_sound = pygame.mixer.Sound(path)
		path = os.path.join('assets/sounds/use_energy.mp3')
		self.use_sound = pygame.mixer.Sound(path)

class Game:
	def __init__(self, hs=0):
		self.bg = [BG(x=0), BG(x=LANE_WIDTH)]
		self.instruction = Instruction()
		self.mc = MC()
		self.enemy = Enemy()
		self.energy_bar = EnergyBar()
		self.obstacles = []
		self.speed = GAME_SPEED
		self.playing = False
		self.set_sound()
		self.set_labels()
		self.spawn_obstacle(0)

	def set_labels(self):
		big_font = pygame.font.SysFont('monospace', 24, bold=True)
		small_font = pygame.font.SysFont('monospace', 18)
		self.dead_lbl = big_font.render(f'G A M E  O V E R', 1, (0, 0, 0))
		self.restart_lbl = small_font.render(f'press up key to restart', 1, (0, 0, 0))
		self.start_lbl = big_font.render(f'press space to start game', 1, (0, 0, 0))

	def show_start_msg(self):
		screen.blit(self.start_lbl, (LANE_WIDTH // 2 - self.start_lbl.get_width() // 2, LANE_HEIGHT // 4))

	def set_sound(self):
		path = os.path.join('assets/sounds/die.wav')
		self.sound = pygame.mixer.Sound(path)

	def start(self):
		self.playing = True

	def over(self):
		self.sound.play()
		screen.blit(self.dead_lbl, (LANE_WIDTH // 2 - self.dead_lbl.get_width() // 2, LANE_HEIGHT // 4))
		screen.blit(self.restart_lbl, (LANE_WIDTH // 2 - self.restart_lbl.get_width() // 2, LANE_HEIGHT // 2))
		self.energy_bar.clear_energy()
		self.enemy.reset_monster()
		self.playing = False

	def tospawn(self, loops):
		return loops % 100 == 0

	def spawn_obstacle(self, loops):
		MIN_GAP = 200
		# list obstacles
		if len(self.obstacles) > 0:
			prev_obstacle = self.obstacles[-1]
			x = random.randint(prev_obstacle.x + self.mc.width + MIN_GAP, LANE_WIDTH + prev_obstacle.x + self.mc.width + MIN_GAP)
		else:
			x = 500

		obstacle = None
		if loops < 600:
			# only spawn bushes at the early stage
			obstacle = Bush(x)
		else:
			obstacle_type = random.randint(0,10)
			if obstacle_type >= 0 and obstacle_type < 6 :
				# create the new bush
				obstacle = Bush(x)
			elif obstacle_type >= 6 and obstacle_type < 8:
				# create the new rock
				obstacle = Rock(x)
			else:
				# create the new fire
				obstacle = Fire(x)
		self.obstacles.append(obstacle)
	
	def reset(self):
		self.__init__()

def get_dominant_expression(deepface_result, au_result):
	DEEPFACE_W = 0.98
	AU_W = 0.02
	target_exprs = ['happy', 'surprise', 'angry', 'neutral']
	final_scores = [] 
	for expr in target_exprs:
		final_scores.append(deepface_result[expr] * DEEPFACE_W + au_result[expr] * AU_W)
	return target_exprs[np.argmax(final_scores)]

def main(game_mode="normal"):
	
	# objects
	game = Game()

	# variables
	loops = 0
	over = False
	force_over = False

	game.show_start_msg()

	# pygame loop
	while True:
		# read webcam images		
		success, frame = cap.read()
		if not success:
			print("Error: failed reading image from webcam")
			break
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		# predict expression
		if loops % 20 == 1 and game.playing and not game.mc.in_action():
			# predict emotion with deepface
			deepface_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
			# print(len(deepface_result))
			# for result in deepface_result:
			# 	print(result['region'])
			if len(deepface_result) == 0:
				continue
			deepface_result = deepface_result[0]['emotion']
			
			# face detection using OpenCV
			faces = face_cascade.detectMultiScale(frame, 1.1, 4)
			if len(faces) == 0:
				continue
			x, y, w, h = faces[0]
			face_img = frame[y:y+h, x:x+w]
			face_img = cv2.resize(face_img, (224, 224))
			face_img = np.expand_dims(face_img, axis=0)
			
			# detect au in face and map to expression
			y_predict = au_detector.predict(face_img, verbose=0)
			ind = np.where(y_predict[1] > 0.8)[1]
			au_result = get_expression_confidence_scores(ind)

			# combine deepface result and au result
			expr = get_dominant_expression(deepface_result, au_result)

			# perform action
			if expr == "happy":
				if game.mc.onground:
					game.mc.jump()
			elif expr == "surprise":
				game.mc.shovel(loops)
			elif expr == "angry" or expr == "disgust":
				if game.energy_bar.value == 5:
					game.mc.dash(loops)
					game.energy_bar.use_energy(loops)
					game.enemy.reduce_monster()
		
		# display webcam image
		frame = cv2.flip(frame, 1) 
		frame = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")
		frame = pygame.transform.scale(frame, (240, 180))
		screen.blit(frame, (0, LANE_HEIGHT))

		# render game assets
		if game.playing:
			for bg in game.bg:
				bg.update(-game.speed)
				bg.show()
		
			game.instruction.show()

			game.mc.update(loops)

			if game.tospawn(loops):
				game.spawn_obstacle(loops)

			for obstacle in game.obstacles:
				obstacle.update(-game.speed)
				obstacle.show()

				# checking collision
				if (not obstacle.hit
					and not obstacle.crash
					and loops % 5 == 1 
					and not over
					and obstacle.check_collision(game.mc)):
					if game.mc.dashing:
						obstacle.get_crash()
					else:
						if (obstacle.__class__.__name__ == 'Bush'):
							game.energy_bar.clear_energy()
							game.enemy.add_monster()
							if (game.enemy.number == 3):
								over=True
						else:
							over=True
			
			game.mc.show()
			game.energy_bar.update(loops)
			game.energy_bar.show()
			game.enemy.update(loops)
			game.enemy.show()
			
			loops += 1

		# normal control using keys/mouse input
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
			
			if event.type == pygame.KEYDOWN:
				if not over:
					if event.key == pygame.K_SPACE:
						if not game.playing: # start game
							game.start()
						elif game.mc.onground: # jump
							game.mc.jump()
					if event.key == pygame.K_LSHIFT: # slide shovel
						game.mc.shovel(loops)
					if event.key == pygame.K_LCTRL and game.energy_bar.value == 5: # dash
						game.mc.dash(loops)
						game.energy_bar.use_energy(loops)
						game.enemy.reduce_monster()

				if event.key == pygame.K_UP: # restart game
					game.reset()
					loops = 0
					over = False
					game.start()
				if event.key == pygame.K_DOWN: # force game over
					over = True
					force_over = True
		
		# game over if not cheating
		if force_over or (game_mode!= "cheat" and over and game.playing):
			game.over()
			force_over = False


		pygame.display.update()
		

# get game mode from command input
game_mode = "cheat" if len(sys.argv) == 2 and sys.argv[1].strip() == "cmpt724" else "normal" 
main(game_mode)

# release resources
cap.release()
pygame.quit()

