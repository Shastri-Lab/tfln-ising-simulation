# Best parameters found so far


## S6

### 3x3 Lattice
Really good results with:
`betas=(0.05, 0.1, 0.2, 0.3), noise_std=0.05`
`solve_hp_isingmachine(model, num_iterations=150_000, num_ics=20, betas=(0.005, 0.0075, 0.01, 0.025), noise_std=0.1)`

## S10
```python
model = load_hp_model_by_name('S10', latdim=(4,3))
solve_hp_isingmachine(model, num_iterations=5_000, num_ics=250, betas=(0.005, 0.004, 0.003), noise_std=0.1)
```


## S30
Meh results:
`solve_hp_isingmachine(model, num_iterations=500, num_ics=10, betas=(0.001, 0.005, 0.01, 0.02), noise_std=0.069)`