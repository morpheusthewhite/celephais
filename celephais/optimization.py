import cplex


def calculate_obj_vector(rooms_seats, lessons_students, rooms_are_rows=False):

    costs = []

    if rooms_are_rows:
        for room_seats in rooms_seats:
            for lesson_students in lessons_students:
                costs.append((lesson_students) / room_seats)
    else:
        for lesson_students in lessons_students:
            for room_seats in rooms_seats:
                costs.append((lesson_students) / room_seats)

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
    print(solution.get_objective_value())

    var_values = solution.get_values(0, cplex_model.variables.get_num() - 1)

    return var_values
