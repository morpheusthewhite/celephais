import cplex

from celephais import metadata

EPSILON = 1


def calculate_obj_vector(rooms_seats, lessons_students, rooms_are_rows=False):
    # the final vector to return
    costs = []

    # coefficients useful for the calculation of costs, stored to speed up the process
    coefficients = []

    # calculate the len(lessons_students) coefficients
    for i in range(len(lessons_students)):

        tmp = 1
        for j, lesson_students in enumerate(lessons_students):
            if not j == i:
                tmp *= (lesson_students - EPSILON * i)

        coefficients.append(tmp)

    if rooms_are_rows:
        for room_seats in rooms_seats:
            for (index, lesson_students) in enumerate(lessons_students):
                costs.append(lesson_students * coefficients[index] / room_seats)
    else:
        for (index, lesson_students) in enumerate(lessons_students):
            for room_seats in rooms_seats:
                costs.append(lesson_students * coefficients[index] / room_seats)

    return costs


def solve_assignment_problem(rooms_seats, lessons_students):
    """
    Solve the assignment problem given a list of rooms with room_seats seats and lessons_students students
    :param rooms_seats: a list containing the seats of the rooms
    :param lessons_students: a list containing the number of students of each lesson
    :return: the solutions matrix
    """
    n_rooms = len(rooms_seats)  # the number of rooms available
    n_students = len(lessons_students)  # the number of lessons

    # calculates the cost matrix
    cost_vector = calculate_obj_vector(rooms_seats, lessons_students)

    # create CPLEX object
    cplex_model = cplex.Cplex()
    cplex_model.set_log_stream(None)
    cplex_model.set_warning_stream(None)
    cplex_model.set_results_stream(None)

    # creating the vector of the decision variables
    y_vars_name = ["y" + str(i) + str(j) for i in range(n_students) for j in range(n_rooms)]
    cplex_model.variables.add(names=y_vars_name,
                              types=[cplex_model.variables.type.binary] * n_students * n_rooms,
                              obj=cost_vector)

    # we want to maximize
    cplex_model.objective.set_sense(cplex_model.objective.sense.maximize)

    cplex_model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["y" + str(i) + str(j) for j in range(n_rooms)], val=[1] * n_rooms)
                  for i in range(n_students)],
        senses=["E"] * n_students,
        rhs=[1] * n_students,
        range_values=[0] * n_students)

    cplex_model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["y" + str(i) + str(j) for i in range(n_students)], val=[1] * n_students)
                  for j in range(n_rooms)],
        senses=["L"] * n_rooms,
        rhs=[1] * n_rooms,
        range_values=[0] * n_rooms)

    cplex_model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["y" + str(i) + str(j)], val=[lessons_students[i]])
                  for i in range(n_students) for j in range(n_rooms)],
        senses=["L"] * n_rooms * n_students,
        rhs=[rooms_seats[j] for i in range(n_students) for j in range(n_rooms)],
        range_values=[0] * n_rooms * n_students)

    try:
        cplex_model.solve()
    except cplex.CplexError as exc:
        print(exc)
        return

    solution = cplex_model.solution

    var_values = solution.get_values(0, cplex_model.variables.get_num() - 1)

    return var_values


def assign_classes(classes, rooms):
    # first group classes by time
    classes_grouped = metadata.group_by_time(classes)

    rooms_cap = list(map(lambda x: x["cap"], rooms))
    n_rooms = len(rooms_cap)

    result = {}

    # solving the problem for each time
    for time, classes in classes_grouped.items():
        students_list = list(map(lambda d: d["students"], classes))

        solution = solve_assignment_problem(rooms_cap, students_list)

        for i in range(len(solution)):
            if solution[i] == 1:
                lesson_index = int(i/n_rooms)
                room_index = i % n_rooms

                class_ = classes[lesson_index]
                class_["room"] = rooms[room_index]["name"]
                try:
                    result[time].append(class_)
                except KeyError:
                    result[time] = [class_]

    return result
