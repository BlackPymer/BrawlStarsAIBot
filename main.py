from random import randint

score = 0
while True:
    print("перед тобой 3 двери")
    print("Выбери одну (1,3)")
    try:
        chosen_door = int(input())
    except ValueError:
        print("некоректный ввод")
        continue
    if chosen_door < 1 or chosen_door > 3:
        print("тупой что ли")
        continue
    ghost_door = randint(1, 3)
    if chosen_door != ghost_door:
        print("повезло ска")
        score += 1
    else:
        break

