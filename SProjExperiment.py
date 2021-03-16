"""
@author: Evan Petratos
created 10/5/2020
"""

from psychopy import visual, core, event #import some libraries from PsychoPy
from psychopy import data
from psychopy.sound import backend_sounddevice
import csv
import time

#sound stimuli
stimuli = ["Gregorian_Chant.wav", "Gregorian_Chant.wav", #each stimuli is repeated
"Mozart1.wav", "Mozart1.wav",
"Mozart2.wav", "Mozart2.wav",
"Beethoven.wav", "Beethoven.wav", "break.wav", #participants' break
"Bruckner1.wav", "Bruckner1.wav",
"Bruckner2.wav", "Bruckner2.wav",
"Bruckner3.wav", "Bruckner3.wav",
"Chopin1.wav", "Chopin1.wav",
"Chopin2.wav", "Chopin2.wav",
"Rochmaninoff.wav", "Rochmaninoff.wav", "break.wav"] #used to show exit message

#global variables
markings = [] #marks times of events
prog_state = [] #used to track the state of the program
sound_state = 0 #used to traverse through stimuli
sound = backend_sounddevice.SoundDeviceSound(stimuli[0])
play = False

#create a window
win = visual.Window([1366,768],monitor="testMonitor", units="deg")
start_msg = visual.TextStim(win, text="Usage:\nPress [p] to begin.\nPress [space] to mark endings of phrases.\nPress [n] then [p] to start the next piece.\nPress [q] to end the experiment.")
break_msg = visual.TextStim(win, text="Break.\nPress [n] then [p] to resume the experiment.")
exit_msg = visual.TextStim(win, text="Press [q] to end experiment.")

#timer created with core.Clock()
timer = core.Clock()

#func to start the timer and present the stimuli
def begin():
    global sound
    global play
    global sound_state
    global startup
    
    if play == True:
        core.wait(10)
    elif play == False:
        play = True
        prog_state.append(0)
        markings.append(sound_state) #shows which stimulus is being marked
        sound = backend_sounddevice.SoundDeviceSound(stimuli[sound_state])
        timer.reset()
        sound.play()
        sound_state += 1

#func to record markings
def mark():
    if len(prog_state)>0: #won't mark until stimuli is presented
        markings.append(timer.getTime())
        print(timer.getTime())

#func to skip the current stimuli
def skip():
    global sound
    global play
    sound.stop()
    play = False

#func to exit window
def cleanup():
    print(markings)

    #write markings to csv file
    with open('marks.csv', 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, delimiter=',')
        wr.writerow(markings)

    #close window and end program
    win.close()
    core.quit()

#event keys used
event.globalKeys.clear()
event.globalKeys.add(key='p', func=begin)
event.globalKeys.add(key='space', func=mark)
event.globalKeys.add(key='n', func=skip)
event.globalKeys.add(key='q', func=cleanup)

#keeps the window running
while True:
    if sound_state < 9:
        start_msg.draw()
        win.flip()
    elif sound_state == 9:
        break_msg.draw()
        win.flip()
    elif sound_state > 9 and sound_state < 21:
        start_msg.draw()
        win.flip()
    else:
        exit_msg.draw()
        win.flip()
