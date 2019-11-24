from dataset import Adolescents, Adults, Children


def main():
    for data_set in [Adolescents, Adults, Children]:
        data_set.fit()
        data_set.score()
        print(data_set.accuracies)


if __name__ == "__main__":
    main()
