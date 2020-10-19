import numpy
import scipy.special

# Определение класса нейронной сети
class neuralNetwork:

	# Инициализация нейронной сети
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# Задаём количество узлов во входном, скрытом и выходном узле
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		# Коэффициент обучения
		self.lr = learningrate

		# Инициализация матрицы весов
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5))

		# Использование сигмоиды в качестве функции активации
		self.activation_function = lambda x: scipy.special.expit(x)
		pass

	# Тренировка нейронной сети
	def train(self, inputs_list, targets_list):
		# Преобразовать список входных значений в двухмерный массив
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T

		# Рассчитать входящие сигналы для скрытого слоя
		hidden_inputs = numpy.dot(self.wih, inputs)
		# Рассчитать исходящие сигналы для скрытого слоя
		hidden_outputs = self.activation_function(hidden_inputs)

		# Рассчитать входящие сигналы для выходного слоя
		final_inputs = numpy.dot(self.who, hidden_outputs)
		# Рассчитать исходящие сигналы для выходного слоя
		final_outputs = self.activation_function(final_inputs)

		# Ошибка = целевое значение - фактическое значение
		output_errors = targets - final_outputs
		# Ошибки скрытого слоя - это ошибки output_errors,
		# распределённые пропорционально весовым коэффициентам связей
		# и рекомбинированные на скрытых узлах

		hidden_errors = numpy.dot(self.who.T, output_errors)

		# Обновить весовые коэффициенты связей между скрытым и выходным слоями
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
										numpy.transpose(hidden_outputs))

		# Обновить весовые коэффициенты связей между входным и скрытым слоями
		self.win += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
										numpy.transpose(inputs))

	# Опрос нейронной сети
	def query(self, inputs_list):
		# Преобразовать список входных значений в двухмерный массив
		inputs = numpy.array(inputs_list, ndmin=2).T

		# Рассчитать входящие сигналы для скрытого слоя
		hidden_inputs = numpy.dot(self.wih, inputs)
		# Рассщитать исходящие сигналы для скрытого слоя
		hidden_outputs = self.activation_function(hidden_inputs)

		# Рассчитать входящие сигналы для выходного слоя
		final_inputs = numpy.dot(self.who, hidden_outputs)
		# Рассчитать исходящие сигналы для выходного слоя
		final_outputs = self.activation_function(final_inputs)

		return final_outputs