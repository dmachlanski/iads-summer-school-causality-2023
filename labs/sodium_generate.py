import numpy as np

def generate_data(n=1000, seed=0, beta1=1.05, binary_treatment=True, binary_cutoff=3.5):
    np.random.seed(seed)
    age = np.random.normal(65, 5, n)
    sodium = age / 18 + np.random.normal(size=n)
    if binary_treatment:
        if binary_cutoff is None:
            binary_cutoff = sodium.mean()
        sodium = (sodium > binary_cutoff).astype(int)
    blood_pressure = beta1 * sodium + 2 * age + np.random.normal(size=n)

    return age, sodium, blood_pressure

if __name__ == "__main__":
    age, sodium, bp = generate_data(10000)
    np.savez('sodium_10k', x=age, t=sodium, y=bp)