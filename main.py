import numpy as np
import random
import time
import matplotlib.pyplot as plt
import sys
import io
import plotly.graph_objs as go


# --- Các hàm từ file .ipynb của bạn ---

# Hàm euclidean_distance, evaluate_fitness, calculate_interception_point,
# generate_population, select_population, crossover, mutate, repair_solution,
# evolutionary_algorithm, process_data, read_data_from_file,
# plot_solution


# Tính khoảng cách Euclidean
def euclidean_distance(point1, point2):
    if point1 is None or point2 is None:
        raise ValueError("Tọa độ điểm không hợp lệ")
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Tính thời gian giao hàng của một lời giải
def evaluate_fitness(solution, coordinates, truck_speed, drone_speed, truck_delivery_time, drone_delivery_time):
    truck_time = 0  # Thời gian xe tải
    drone_time = 0  # Thời gian drone
    truck_pos = coordinates[0]  # Xe tải bắt đầu tại depot (node 0)
    drone_pos = coordinates[0]  # Drone bắt đầu tại depot (node 0)

    for node in solution:
        if node is None or not isinstance(node, tuple) or len(node) != 2:
            continue  # Bỏ qua phần tử không hợp lệ

        customer, delivery_type = node

        if customer < 0 or customer >= len(coordinates):
            continue  # Bỏ qua nếu node vượt ngoài phạm vi tọa độ

        customer_pos = coordinates[customer]

        if delivery_type == "truck":
            # Xe tải di chuyển đến điểm giao hàng
            travel_time = euclidean_distance(truck_pos, customer_pos) / truck_speed
            truck_time += travel_time + truck_delivery_time
            truck_pos = customer_pos

        elif delivery_type == "drone":
            # Tính toán vị trí chặn
            interception_point, interception_time = calculate_interception_point(
                truck_pos, customer_pos, truck_speed, drone_speed, truck_delivery_time, drone_delivery_time  # Thêm thời gian giao hàng
            )

            if interception_point is not None:
                # Drone di chuyển đến điểm giao hàng và quay lại vị trí chặn
                drone_time += euclidean_distance(drone_pos, customer_pos) / drone_speed
                drone_time += drone_delivery_time
                drone_time += interception_time
                drone_pos = interception_point
            else:
                # Không thể chặn, xe tải chờ drone tại node tiếp theo
                drone_time += euclidean_distance(drone_pos, customer_pos) / drone_speed + drone_delivery_time
                drone_pos = customer_pos

        # Cập nhật thời gian chậm nhất giữa xe tải và drone
        truck_time = max(truck_time, drone_time)

    return 1 / truck_time  # Fitness: thời gian càng ngắn, fitness càng cao

# Tính vị trí chặn (đã được cập nhật)
def calculate_interception_point(c_i, c_j, v_t, v_d, truck_delivery_time, drone_delivery_time):
    """
    Tính toán điểm chặn giữa xe tải và drone, có tính đến thời gian giao hàng.

    Args:
      c_i: Tọa độ của điểm hiện tại của xe tải.
      c_j: Tọa độ của điểm giao hàng của drone.
      v_t: Vận tốc của xe tải.
      v_d: Vận tốc của drone.
      truck_delivery_time: Thời gian giao hàng của xe tải tại điểm c_i.
      drone_delivery_time: Thời gian giao hàng của drone tại điểm c_j.

    Returns:
      Một tuple (p_Ij, t_Ij) trong đó:
        p_Ij: Tọa độ của điểm chặn.
        t_Ij: Thời gian để drone bay từ điểm hiện tại đến điểm chặn.
      Trả về (None, None) nếu không thể chặn.
    """
    if c_i is None or c_j is None:
        raise ValueError("Tọa độ điểm không hợp lệ")

    delta = np.array(c_j) - np.array(c_i)
    d_ij = np.linalg.norm(delta)  # Khoảng cách Euclidean

    # Tính thời gian drone bay thẳng đến điểm c_j
    t_direct = d_ij / v_d + drone_delivery_time

    # Tính thời gian xe tải di chuyển đến điểm c_j
    t_truck = d_ij / v_t + truck_delivery_time

    # Nếu drone đến c_j trước xe tải thì không cần chặn
    if t_direct <= t_truck:
        return None, None

    a = v_d**2 - v_t**2
    b = -2 * np.dot(delta, [v_t, v_t])
    c = -d_ij**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None, None  # Không có nghiệm thực, không thể chặn

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    # Lọc các nghiệm hợp lệ (t > 0)
    valid_times = [t for t in (t1, t2) if t > 0]

    if not valid_times:  # Nếu không có nghiệm hợp lệ
        return None, None

    t_Ij = min(valid_times)  # Lấy nghiệm nhỏ nhất trong các nghiệm hợp lệ

    # Kiểm tra xem drone có đến điểm chặn trước khi xe tải đến điểm c_j không
    if t_Ij + drone_delivery_time >= t_truck:
        return None, None # Không thể chặn

    p_Ij = np.array(c_i) + v_t * t_Ij
    return p_Ij, t_Ij

# Tạo quần thể ban đầu
def generate_population(pop_size, num_customers):
    population = []
    for _ in range(pop_size):
        # Đảm bảo mỗi khách hàng chỉ xuất hiện một lần
        customers = list(range(1, num_customers + 1))
        random.shuffle(customers)
        solution = [(customer, random.choice(["truck", "drone"])) for customer in customers]
        population.append(solution)
    return population

# Hàm chọn lọc (có elitism)
def select_population(population, fitness, elite_size):
    sorted_population = sorted(population, key=lambda x: fitness.get(tuple(x), 0), reverse=True)
    selected = sorted_population[:elite_size]  # Giữ lại elite_size cá thể tốt nhất
    remaining_population = sorted_population[elite_size:]
    probabilities = [fitness.get(tuple(x), 0) for x in remaining_population]
    probabilities = np.array(probabilities) / sum(probabilities) if sum(probabilities) > 0 else np.ones(len(remaining_population)) / len(remaining_population)
    while len(selected) < len(population):
        selected.append(random.choices(remaining_population, probabilities)[0])
    return selected

# Lai ghép (PMX)
def crossover(parent_a, parent_b):
    size = len(parent_a)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size

    # Copy đoạn con từ parent_a sang child
    child[start:end] = parent_a[start:end]

    # Mapping các node trong đoạn con
    mapping = {parent_a[i]: parent_b[i] for i in range(start, end)}

    # Điền các node còn lại vào child
    for i in range(size):
        if child[i] is None:
            node = parent_b[i]
            while node in mapping:
                node = mapping[node]
            child[i] = node

    return child

# Đột biến
def mutate(solution, mutation_rate):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            # Đổi loại giao hàng (truck/drone)
            customer, delivery_type = solution[i]
            solution[i] = (customer, "truck" if delivery_type == "drone" else "drone")
    return solution

# Sửa lỗi: đảm bảo mỗi khách hàng chỉ xuất hiện 1 lần
def repair_solution(solution, num_customers):
    customers_visited = set()
    new_solution = []
    previous_delivery_type = None

    for customer, delivery_type in solution:
        if customer not in customers_visited:
            if delivery_type == "drone" and previous_delivery_type == "drone":
                delivery_type = "truck"  # Change to truck if the previous customer was also delivered by drone
            new_solution.append((customer, delivery_type))
            customers_visited.add(customer)
            previous_delivery_type = delivery_type

    # Add missing customers
    missing_customers = set(range(1, num_customers + 1)) - customers_visited
    for customer in missing_customers:
        if previous_delivery_type == "drone":
            delivery_type = "truck"
        else:
            delivery_type = random.choice(["truck", "drone"])
        new_solution.append((customer, delivery_type))
        customers_visited.add(customer)
        previous_delivery_type = delivery_type

    return new_solution

# Chạy thuật toán tiến hóa (đã được cập nhật)
def evolutionary_algorithm(coordinates, truck_speed, drone_speed, truck_delivery_time, drone_delivery_time, population_size, num_generations, mutation_rate, elite_size):
    num_customers = len(coordinates) - 1
    population = generate_population(population_size, num_customers)
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(num_generations):
        fitness = {tuple(sol): evaluate_fitness(sol, coordinates, truck_speed, drone_speed, truck_delivery_time, drone_delivery_time) for sol in population}
        best_gen_solution = max(population, key=lambda x: fitness.get(tuple(x), 0))
        best_gen_fitness = fitness.get(tuple(best_gen_solution), 0)

        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            best_solution = best_gen_solution

        selected_population = select_population(population, fitness, elite_size)
        next_population = []

        for i in range(0, len(selected_population), 2):
            parent_a = selected_population[i]
            parent_b = selected_population[(i + 1) % len(selected_population)]
            child = crossover(parent_a, parent_b)
            child = mutate(child, mutation_rate)
            child = repair_solution(child, num_customers)
            next_population.append(child)

        population = next_population

    return best_solution, 1 / best_fitness


# ... (Sao chép tất cả các hàm từ file .ipynb của bạn vào đây) ...
def process_data(dataset_str):
    """
    Xử lý string dữ liệu đầu vào và trả về một dictionary chứa thông tin về bài toán.

    Args:
        dataset_str: String dữ liệu đầu vào.

    Returns:
        Một dictionary chứa các thông tin sau:
            truck_speed: Vận tốc của xe tải.
            drone_speed: Vận tốc của drone.
            num_nodes: Số lượng nút giao hàng.
            coordinates: Danh sách tọa độ của các nút, bao gồm cả depot.
            truck_delivery_time: Thời gian giao hàng của xe tải (mặc định là 1).
            drone_delivery_time: Thời gian giao hàng của drone (mặc định là 1).
    """
    lines = dataset_str.strip().split('\n')
    data = {}

    try:
        data["truck_speed"] = float(lines[1])
        data["drone_speed"] = float(lines[3])
        data["num_nodes"] = int(lines[5])
    except ValueError as e:
        print(f"Lỗi khi xử lý thông tin chung: {e}")
        return None  # Trả về None nếu không thể đọc thông tin chung

    data["coordinates"] = []
    data["truck_delivery_time"] = 0  # Mặc định
    data["drone_delivery_time"] = 0  # Mặc định

    # Xử lý tọa độ depot
    depot_line = lines[7].split()
    try:
        x, y = float(depot_line[0]), float(depot_line[1])
        data["coordinates"].append((x, y))  # Thêm tọa độ depot vào danh sách
    except ValueError as e:
        print(f"Lỗi khi xử lý dòng depot: {lines[7]}. Lỗi: {e}")
        return None

    # Xử lý tọa độ các điểm giao hàng khác
    for line in lines[8:]:
        parts = line.split()
        try:
            if len(parts) >= 3:
                x, y = float(parts[0]), float(parts[1])
                data["coordinates"].append((x, y))
        except ValueError as e:
            print(f"Lỗi khi xử lý dòng: {line}. Lỗi: {e}")
            continue

    return data

    
def read_data_from_file(file_path):
  """Đọc dữ liệu từ file và xử lý bằng hàm process_data.

  Args:
    file_path: Đường dẫn đến file dữ liệu.

  Returns:
    Một dictionary chứa thông tin về bài toán,
    hoặc None nếu có lỗi xảy ra.
  """
  try:
    with open(file_path, 'r') as file:
      data_str = file.read()
    return process_data(data_str)
  except FileNotFoundError:
    print(f"Không tìm thấy file: {file_path}")
    return None
  except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
    return None

def plot_solution(coordinates, solution):
    """Vẽ chu trình tốt nhất trên đồ thị, hiển thị nhãn của các node và mũi tên ở giữa."""
    
    # Lấy tọa độ x và y của tất cả các điểm
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    # Vẽ các điểm giao hàng
    customers_trace = go.Scatter(
        x=x_coords[1:], y=y_coords[1:], mode='markers', marker=dict(color='blue'), name='Customers'
    )

    # Vẽ depot
    depot_trace = go.Scatter(
        x=[x_coords[0]], y=[y_coords[0]], mode='markers', marker=dict(color='red', symbol='square'), name='Depot'
    )

    truck_path = [0]  # Bắt đầu từ depot
    drone_path = [0]  # Bắt đầu từ depot

    for node, delivery_type in solution:
        if delivery_type == "truck":
            truck_path.append(node)
            drone_path.append(node)
        else:
            drone_path.append(node)

    truck_path.append(0)  # Quay lại depot
    drone_path.append(0)  # Quay lại depot

    # Vẽ đường đi của xe tải
    truck_x = [x_coords[i] for i in truck_path]
    truck_y = [y_coords[i] for i in truck_path]
    truck_trace = go.Scatter(
        x=truck_x, y=truck_y, mode='lines+markers', line=dict(color='green'), name='Truck Route'
    )

    # Vẽ đường đi của drone
    drone_x = [x_coords[i] for i in drone_path]
    drone_y = [y_coords[i] for i in drone_path]
    drone_trace = go.Scatter(
        x=drone_x, y=drone_y, mode='lines+markers', line=dict(color='orange', dash='dash'), name='Drone Route'
    )

    # Hiển thị nhãn của node
    annotations = [
        dict(
            x=x_coords[i], y=y_coords[i], text=str(i), showarrow=False, font=dict(color='black', size=10)
        ) for i in range(len(coordinates))
    ]

    for i in range(1, len(truck_path)):
        x_start, y_start = x_coords[truck_path[i-1]], y_coords[truck_path[i-1]]
        x_end, y_end = x_coords[truck_path[i]], y_coords[truck_path[i]]
        annotations.append(
            dict(
                x=(x_start + x_end) / 2, y=(y_start + y_end) / 2,
                axref='x', ayref='y',
                ax=x_start, ay=y_start,
                xref='x', yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='green'
            )
        )

    for i in range(1, len(drone_path)):
        x_start, y_start = x_coords[drone_path[i-1]], y_coords[drone_path[i-1]]
        x_end, y_end = x_coords[drone_path[i]], y_coords[drone_path[i]]
        annotations.append(
            dict(
                x=(x_start + x_end) / 2, y=(y_start + y_end) / 2,
                axref='x', ayref='y',
                ax=x_start, ay=y_start,
                xref='x', yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='orange'
            )
        )


    layout = go.Layout(
        title='Best Solution Route',
        xaxis=dict(title='X Coordinate'),
        yaxis=dict(title='Y Coordinate'),
        showlegend=True,
        annotations=annotations
    )

    figure = go.Figure(data=[customers_trace, depot_trace, truck_trace, drone_trace], layout=layout)
    return figure


# --- Hàm run_algorithm ---
import time

def run_algorithm(data, population_size, num_generations, mutation_rate, elite_size):
    """
    Chạy thuật toán tiến hóa trên dữ liệu đầu vào và trả về kết quả.

    Args:
        data (dict): Dữ liệu đầu vào đã được xử lý.
        population_size (int): Kích thước quần thể.
        num_generations (int): Số thế hệ.
        mutation_rate (float): Tỷ lệ đột biến.
        elite_size (int): Kích thước nhóm ưu tú.

    Returns:
        tuple: (best_solution, best_time, execution_time)
    """
    if data:
        coordinates = data["coordinates"]
        truck_speed = data["truck_speed"]
        drone_speed = data["drone_speed"]
        truck_delivery_time = data["truck_delivery_time"]
        drone_delivery_time = data["drone_delivery_time"]

        start_time = time.time()
        best_solution, best_time = evolutionary_algorithm(
            coordinates, truck_speed, drone_speed, truck_delivery_time, drone_delivery_time,
            population_size, num_generations, mutation_rate, elite_size
        )
        end_time = time.time()
        execution_time = end_time - start_time

        return best_solution, best_time, execution_time
    else:
        return None, None, None

# --- Hàm main ---

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    file_path = 'data\\uniform-3-n19.txt'  # Thay đổi đường dẫn file nếu cần
    data = read_data_from_file(file_path)

    if data:
        print("Dữ liệu từ file:")
        for key, value in data.items():
            print(f"{key}: {value}")

        population_size = 200
        num_generations = 10000
        mutation_rate = 0.3
        elite_size = 20
        coordinates = data["coordinates"]
        truck_speed = data["truck_speed"]
        drone_speed = data["drone_speed"]
        truck_delivery_time = data["truck_delivery_time"]
        drone_delivery_time = data["drone_delivery_time"]

        start_time = time.time()
        best_solution, best_time = evolutionary_algorithm(
            coordinates, truck_speed, drone_speed, truck_delivery_time, drone_delivery_time,
            population_size, num_generations, mutation_rate, elite_size
        )
        end_time = time.time()
        execution_time = end_time - start_time

        print("Thời gian chạy:", execution_time, "giây")
        print("Best Solution:", best_solution)
        print("Best Delivery Time:", best_time)

        plot_solution(coordinates, best_solution)
        plt.show()

if __name__ == "__main__":
    main()