import numpy as np

class RandomHouse(object):
    """Random house generator

    This class generates a random house with various rooms and
    receiver sensors, and allows to generate walk-arounds with
    corresponding RSSI values per receiver. The generated house
    layout may be impossible, as the distances, and connections
    between rooms does not have constrains during the generation.

    Parameters
    ----------
    n_rooms : int, default=7
        Number of rooms

    largest_distance : int, default=40
        Maximum distance between different rooms

    min_distance : int, default=5
        Minimum distance between different rooms

    min_size : int, default=3
        Minimum width/lenght of a room

    max_size : int, default=7
        Maximum width/lenght of a room

    n_sensors : int, default=4
        Number of sensors (must be smaller than number of rooms)

    random_seed : int or None
        If a random_seed is provided, the house layout and the rest of
        parameters will allways generate the same data


    Attributes
    ----------
    walkaround_dict : dictionary of walkarounds
        A walkaround is a dictionary with one key per room_name and the following values
            locations: matrix (n_locations, 2)
                Several locations inside the room with x and y coordinates
                with respect to the south-west room
            straight_path: matrix (n_locations*samples_between_locations, 2)
                Contains linear interpolations between each location visited in the
                room in x, y coordinates with respect to the south-west corner of the
                room.
            path: matrix (n_locations*samples_between_locations, 2)
                Corresponds to the straight path with added noise in the coordinates
                simulating movement of the RSSI emisor.
    room_names: list of strings
        Name of each room of the house (eg. bedroom, bedroom2, bathroom, kitchen)

    sensor_locations: list of strings
        Name of each room that contains a sensor receiver

    sensor_constant_loss: list of ints
        Randomly generated list of losses in dB for each sensor. This simulates
        imperfections on each receiver as well as other losses given its location

    room_dimesions: list of tuples
        One tuple per room containing width and length of the room

    distance_m: matrix (n_rooms, n_rooms)
        Matrix with distances between each pair of rooms symettric agains the diagonal
        and with zeros in the diagonal. These are not walking distances, but in a
        straight line.

    adjacency_m: matrix (n_rooms, n_rooms)
        Matrix indicating connections between each pair of rooms

    walls_loss_m: matrix (n_rooms, n_rooms)
        Matrix with loss in dB between each pair of rooms, simulating different
        properties of the walls (eg. thickness, material, objects).

    number_of_experiments: int
        Number of walkarounds retrieved/generated

    Examples
    --------
    >>> house = RandomHouse(n_rooms=7, random_seed=42)
    >>> walkaround = house.retrieve_walkaround(number=0)
    >>> X_rssi, y = house.get_sensor_rssi(number=0)
    """
    def __init__(self, n_rooms=7, largest_distance=40, min_distance=10,
                 min_size=3, max_size=7, n_sensors=4, random_seed=None):
        self.n_rooms = n_rooms
        self.largest_distance = largest_distance
        self.random_seed = random_seed
        self.n_sensors = n_sensors
        self.walkaround_dict = {}

        self.room_names = self.generate_room_names()
        self.sensor_locations = self.generate_sensor_locations()
        self.sensor_constant_loss = self.generate_sensor_constant_losses()
        self.room_dimensions = self.generate_room_dimensions(min_size, max_size)
        self.distance_m = self.generate_distance_matrix()
        self.adjacency_m = self.distance_m <= min_distance
        self.walls_loss_m = self.generate_walls_loss_matrix()


    @property
    def number_of_experiments(self):
        return len(self.walkaround_dict)

    def generate_room_names(self):
        np.random.seed(self.random_seed)
        basic_rooms = ['bedroom', 'livingroom', 'bathroom', 'kitchen']
        additional_rooms = ['bedroom', 'hall', 'stairs', 'corridor', 'toilet', 'garage', 'study']

        n_basic_rooms = self.n_rooms if self.n_rooms < len(basic_rooms) else len(basic_rooms)
        n_addit_rooms = max([0, self.n_rooms - n_basic_rooms])

        room_names = list(np.random.choice(basic_rooms, n_basic_rooms, replace=False))
        for new_room in np.random.choice(additional_rooms, n_addit_rooms, replace=True):
            suffix = ''
            i = 1
            while new_room + suffix in room_names:
                i += 1
                suffix = str(i)
            room_names.append(new_room + suffix)
        return room_names

    def generate_room_dimensions(self, min_size=2, max_size=8):
        np.random.seed(self.random_seed)
        return np.random.randint(min_size, max_size, size=(self.n_rooms, 2))

    def generate_sensor_locations(self):
        np.random.seed(self.random_seed)
        return np.random.choice(self.room_names, self.n_sensors, replace=False)

    def generate_sensor_constant_losses(self):
        np.random.seed(self.random_seed)
        return np.random.randint(2, 10, self.n_sensors)

    def generate_distance_matrix(self):
        np.random.seed(self.random_seed)
        distance_m = np.triu(np.random.randint(1, self.largest_distance,
                                               size=(self.n_rooms, self.n_rooms)), +1)
        i_lower = np.tril_indices(self.n_rooms, -1)
        distance_m[i_lower] = distance_m.T[i_lower]
        return distance_m

    def generate_walls_loss_matrix(self):
        np.random.seed(self.random_seed)
        walls_m = np.triu(np.random.randint(2, 12,
                                            size=(self.n_rooms, self.n_rooms)), +1)
        i_lower = np.tril_indices(self.n_rooms, -1)
        walls_m[i_lower] = walls_m.T[i_lower]
        return walls_m

    def retrieve_walkaround(self, number=0, n_locations=5, samples_inbetween=10):
        if (number in self.walkaround_dict) and \
           (self.walkaround_dict[number][self.room_names[0]]['path'].shape[0] == n_locations*samples_inbetween):
            return self.walkaround_dict[number]

        self.walkaround_dict[number] = {}
        for i, (room_name, room_size) in enumerate(zip(self.room_names, self.room_dimensions)):
            if (self.random_seed is not None) and (number is not None):
                random_seed = self.random_seed + number + i
            else:
                random_seed = None
            walkaround = {}
            walkaround['locations'] = self.get_locations(room_size, n_locations, random_seed=random_seed)
            walkaround['straight_path'] = self.get_fine_locations(walkaround['locations'],
                                                                  inbetween=samples_inbetween)
            walkaround['path'] = self.add_walking_noise(walkaround['straight_path'], room_size,
                                                         random_seed=random_seed)
            self.walkaround_dict[number][room_name] = walkaround
        return self.walkaround_dict[number]

    def walkaround_as_x_y(self, number, key):
        X = []
        y = []
        for i, (room_name, walkaround) in enumerate(self.walkaround_dict[number].items()):
            X.append(walkaround[key])
            y.append(np.ones(walkaround[key].shape[0], dtype=int)*i)
        X = np.vstack(X)
        y = np.hstack(y).astype(int)
        return X, y

    def get_sensor_rssi(self, number, key='path', min_value=-120):
        def path_loss_model(distances, environment_loss, system_loss):
            return -10*environment_loss*np.log10(distances) - system_loss

        X_dist, y = self.get_sensor_distances(number, key)
        X_rssi = []
        for i in range(self.n_sensors):
            if self.random_seed is not None:
                np.random.seed(self.random_seed+i)
            environment_loss = np.abs(np.random.randn(X_dist.shape[0]))*3 + self.walls_loss_m[i, y]
            current_rssi = path_loss_model(X_dist[:,i], environment_loss,
                                           self.sensor_constant_loss[i])
            current_rssi[current_rssi < -np.random.randint(50, -min_value)] = np.nan
            X_rssi.append(current_rssi)
        X_rssi = np.vstack(X_rssi).T
        return X_rssi, y

    def get_sensor_distances(self, number, key='path'):
        X, y = self.walkaround_as_x_y(number, key)
        eucl_distances = np.sqrt(np.sum(X**2, axis=1))
        X_dist = []
        for i in range(self.n_sensors):
            distances = eucl_distances + self.distance_m[i][y]
            X_dist.append(distances)
        X_dist = np.vstack(X_dist).T
        return X_dist, y

    @staticmethod
    def get_locations(room_size, n_locations, random_seed=None):
        np.random.seed(random_seed)
        return np.vstack([np.random.uniform(0, max_size, size=n_locations)
                                        for max_size in room_size]).T
    @staticmethod
    def get_fine_locations(locations, inbetween=10):
        long_locations = [locations[0].reshape(1, -1),]
        for i, current in enumerate(locations, 1):
            along_x = np.linspace(long_locations[-1][-1][0], current[0], inbetween)
            along_y = np.linspace(long_locations[-1][-1][1], current[1], inbetween)
            long_locations.append(np.vstack((along_x,along_y)).T)
        return np.vstack(long_locations)

    @staticmethod
    def add_walking_noise(locations, room_size, random_seed=None):
        np.random.seed(random_seed)
        locations += np.random.randn(locations.shape[0], locations.shape[1])/6
        locations = np.clip(locations, (0+0.3, 0+0.3), (room_size[0]-0.3, room_size[1]-0.3))
        return locations
