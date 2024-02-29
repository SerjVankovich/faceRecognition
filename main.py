import os

from PIL import Image
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def load_dataset():
    file_list = os.listdir('dataset')
    classes = []
    for i in range(1, 42):
        files_of_class = np.array(
            [process_image(filename) for filename in file_list if filename[:-4].split('_')[1] == str(i)])
        classes.append(files_of_class)
    return np.array(classes)


def process_image(filename):
    image = Image.open("dataset/" + filename)
    if image.mode == "RGB":
        image = image.convert("P", palette=Image.ADAPTIVE, colors=8)
    return np.array(image, dtype="float64").flatten('F')


def pca(images: np.ndarray):
    mean_image = images.mean(axis=0)
    centered_images = (images - mean_image).T
    covariance_matrix = centered_images.T.dot(centered_images)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    eigen_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigen_pairs.sort(key=lambda pair: pair[0], reverse=True)
    U = centered_images.dot(np.array([eigen_pair[1] for eigen_pair in eigen_pairs[:35]]).T)
    return U, mean_image


def lda(imgs_per_class: np.ndarray):
    all_imgs = []
    mean_class_images = []
    qs = []
    train_data, train_labels = get_train_data(imgs_per_class, len(imgs_per_class[0]))
    eigenfaces, mean_face = pca(train_data)
    imgs_per_class = np.array(
        [eigenfaces.T.dot((image - mean_face).T).T for image in [images for images in imgs_per_class]])

    for imgs in imgs_per_class:
        for image in imgs:
            all_imgs.append(image)
        mean_class_images.append(imgs.mean(axis=0))
        qs.append(len(imgs))

    all_imgs = np.array(all_imgs)
    mean_class_images = np.array(mean_class_images)
    qs = np.array(qs)

    mean_image = all_imgs.mean(axis=0)
    sb_dim = mean_image.shape[0]
    s_b = np.zeros((sb_dim, sb_dim))
    s_w = np.zeros((sb_dim, sb_dim))

    for i in range(len(mean_class_images)):
        centered = (mean_class_images[i] - mean_image).reshape(1, -1)
        s_b += qs[i] * (centered.T.dot(centered))
        for image in imgs_per_class[i]:
            centered_mean = (image - mean_class_images[i]).reshape(1, -1)
            temp = centered_mean.T.dot(centered_mean)
            s_w += temp

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
    eigen_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigen_pairs.sort(key=lambda pair: pair[0], reverse=True)
    vectors = np.array([eigen_pair[1] for eigen_pair in eigen_pairs]).T
    return eigenfaces.dot(vectors), eigenfaces.dot(mean_image)


def get_train_data(dataset, num_of_samples):
    train_data = []
    train_labels = []
    for i in range(len(dataset)):
        for image in dataset[i][:num_of_samples]:
            train_data.append(image)
        train_labels += ([i] * num_of_samples)
    return np.array(train_data), np.array(train_labels)


def get_test_data(dataset, num_of_samples):
    test_data = []
    test_labels = []
    for i in range(len(dataset)):
        for image in dataset[i][-num_of_samples:]:
            test_data.append(image)
        test_labels += ([i] * num_of_samples)
    return np.array(test_data), np.array(test_labels)


def show_eigenfaces(eigenfaces):
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
    # Show the first 16 eigenfaces
    for i in range(16):
        axes[i % 4][i // 4].imshow(eigenfaces.T[i].reshape(70, 80).T, cmap="gray")
    plt.show()


def pca_test(train_data, train_data_labels, test_data, test_labels, eigenfaces, mean_face):
    weights = eigenfaces.T.dot((train_data - mean_face).T)
    predict(test_data, test_labels, train_data_labels, mean_face, eigenfaces, weights)


def show_image(im, im_true):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].imshow(im.reshape(70, 80).T, cmap="gray")
    axes[0].set_title("Restored image")
    axes[1].imshow(im_true.reshape(70, 80).T, cmap="gray")
    axes[1].set_title("Original image")
    plt.show()


def predict(test_data, test_labels, train_data_labels, mean_face, eigenfaces, weights):
    y_pred, y_true = [], []
    for i, query in enumerate(test_data):
        query = query.reshape(1, -1)
        query_weight = eigenfaces.T.dot((query - mean_face).T)
        euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
        best_match = np.argmin(euclidean_distance)
        y_pred.append(train_data_labels[best_match])
        y_true.append(test_labels[i])
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("F1:", f1_score(y_true, y_pred, average='macro'))


def lda_test(train_data, train_data_labels, test_data, test_labels, eigenfaces, mean_face):
    weights = eigenfaces.T.dot((train_data - mean_face).T)
    predict(test_data, test_labels, train_data_labels, mean_face, eigenfaces, weights)


def lda_on_mean(train_dataset, test_data, test_labels, eigenfaces, mean_face):
    weights = get_mean_class_images(train_dataset, eigenfaces, mean_face)
    predict(test_data, test_labels, list(range(len(weights))), mean_face, eigenfaces, weights.T)


def pca_on_mean(train_dataset, test_data, test_labels, eigenfaces, mean_face):
    weights = get_mean_class_images(train_dataset, eigenfaces, mean_face)
    predict(test_data, test_labels, list(range(len(weights))), mean_face, eigenfaces, weights.T)


def get_mean_class_images(train_dataset, fisher_faces, mean_face):
    result_messages = []
    for cls in train_dataset:
        result_messages.append(fisher_faces.T.dot((cls - mean_face).mean(axis=0)))
    return np.array(result_messages)


def lda_k_nearest(train_data, train_data_labels, test_data, test_labels, eigenfaces, mean_face):
    weights = eigenfaces.T.dot((train_data - mean_face).T)
    test_data_proj = eigenfaces.T.dot((test_data - mean_face).T)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(weights.T, train_data_labels)
    prediction = neigh.predict(test_data_proj.T)
    print("Precision:", precision_score(test_labels, prediction, average='macro', zero_division=1))
    print("Recall:", recall_score(test_labels, prediction, average='macro', zero_division=1))
    print("F1:", f1_score(test_labels, prediction, average='macro', zero_division=1))


def lda_tree(train_data, train_data_labels, test_data, test_labels, eigenfaces, mean_face):
    weights = eigenfaces.T.dot((train_data - mean_face).T)
    test_data_proj = eigenfaces.T.dot((test_data - mean_face).T)
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(weights.T, train_data_labels)
    prediction = tree.predict(test_data_proj.T)
    print("Precision:", precision_score(test_labels, prediction, average='macro', zero_division=1))
    print("Recall:", recall_score(test_labels, prediction, average='macro', zero_division=1))
    print("F1:", f1_score(test_labels, prediction, average='macro', zero_division=1))


def lda_nn(train_data, train_data_labels, test_data, test_labels, eigenfaces, mean_face):
    weights = eigenfaces.T.dot((train_data - mean_face).T)
    test_data_proj = eigenfaces.T.dot((test_data - mean_face).T)
    mlp = MLPClassifier(max_iter=2000, learning_rate='adaptive', random_state=42)
    mlp.fit(weights.T, train_data_labels)
    prediction = mlp.predict(test_data_proj.T)
    print("Precision:", precision_score(test_labels, prediction, average='macro', zero_division=1))
    print("Recall:", recall_score(test_labels, prediction, average='macro', zero_division=1))
    print("F1:", f1_score(test_labels, prediction, average='macro', zero_division=1))


def pca_k_nearest(train_data, train_data_labels, test_data, test_labels, eigenfaces, mean_face):
    weights = eigenfaces.T.dot((train_data - mean_face).T)
    test_data_proj = eigenfaces.T.dot((test_data - mean_face).T)
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(weights.T, train_data_labels)
    prediction = neigh.predict(test_data_proj.T)
    print("Precision:", precision_score(test_labels, prediction, average='macro', zero_division=1))
    print("Recall:", recall_score(test_labels, prediction, average='macro', zero_division=1))
    print("F1:", f1_score(test_labels, prediction, average='macro', zero_division=1))


def main():
    dataset = load_dataset()
    train_dataset = dataset[:, :7, :]
    train_data, train_data_labels = get_train_data(dataset, 7)
    test_data, test_labels = get_test_data(dataset, 3)
    eigenfaces, mean_face = pca(train_data)
    fisherfaces, mean_fisher_face = lda(train_dataset)

    # Uncomment to see graphics of faces
    # face_to_show = 17
    #
    # weights = eigenfaces.T.dot((train_data - mean_face).T)
    # weight = weights.T[face_to_show]
    # reverse_im = eigenfaces.dot(weight) + mean_face
    # show_image(reverse_im, train_data[face_to_show])
    #
    # show_eigenfaces(eigenfaces)
    # show_eigenfaces(fisherfaces)

    print("CLASSIC PCA:")
    pca_test(train_data, train_data_labels, test_data, test_labels, eigenfaces, mean_face)
    print("\n\nMEAN PCA:")
    pca_on_mean(train_dataset, test_data, test_labels, eigenfaces, mean_face)
    print("\n\nCLASSIC LDA:")
    lda_test(train_data, train_data_labels, test_data, test_labels, fisherfaces, mean_fisher_face)
    print("\n\nMEAN LDA:")
    lda_on_mean(train_dataset, test_data, test_labels, fisherfaces, mean_fisher_face)
    print("\n\nK-MEANS PCA:")
    pca_k_nearest(train_data, train_data_labels, test_data, test_labels, eigenfaces, mean_face)
    print("\n\nK-MEANS LDA:")
    lda_k_nearest(train_data, train_data_labels, test_data, test_labels, fisherfaces, mean_fisher_face)
    print("\n\nTREE LDA:")
    lda_tree(train_data, train_data_labels, test_data, test_labels, fisherfaces, mean_fisher_face)
    print("\n\nMLP LDA:")
    lda_nn(train_data, train_data_labels, test_data, test_labels, fisherfaces, mean_fisher_face)


if __name__ == '__main__':
    main()
