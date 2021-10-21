import yaml
def main():
    with open("config.yaml", "r") as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
        print(data)

if __name__ == "__main__":
    main()