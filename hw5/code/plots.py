import matplotlib.pyplot as plt
import pandas as pd
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

	fig = plt.figure(figsize=(10, 5))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	tl, = ax1.plot(train_losses)
	vl, = ax1.plot(valid_losses)
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Losses')
	ax1.legend([tl, vl], ['Training Loss', 'Validation Loss'])
	ta, = ax2.plot(train_accuracies)
	va, = ax2.plot(valid_accuracies)
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Accuracy')
	ax2.legend([ta, va], ['Training Accuracy', 'Validation Accuracy'])
	# plt.savefig('Learning_curve.png')
	plt.show()
	pass


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

	results_df = pd.DataFrame(results)
	results_df.columns = ['True', 'Pred']
	group = results_df.groupby("True")
	data = []
	for i in range(len(class_names)):
		results_sort = group.get_group(i)
		temp = []
		for j in range(len(class_names)):
			pred_count = results_sort.loc[results_sort.Pred == j].shape[0]
			accuracy = round(pred_count/(results_sort.shape[0]), 2)
			temp.append(accuracy)
		data.append(temp)

	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111)
	ax.set_yticks(range(len(class_names)))
	ax.set_yticklabels(class_names)
	ax.set_xticks(range(len(class_names)))
	ax.set_xticklabels(class_names, rotation=300)
	# ax.xticks(rotation=300)
	im = ax.imshow(data, cmap=plt.cm.get_cmap('YlGnBu'))
	for i in range(len(class_names)):
		for j in range(len(class_names)):
			text = ax.text(j, i, data[i][j],
						   ha="center", va="center")
	plt.colorbar(im)
	plt.title("Normalized Confusion Matrix")
	# plt.savefig('Normalized Confusion Matrix.png')
	plt.show()
	pass
