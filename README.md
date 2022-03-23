
 
# Project Repository

**The link to the video is:?**

## Task I: Build and populate necessary tables

Here is the infrastructure of the table: 

![infrastructure](docs/infrastructure.png)

**Constraints:**
1. ```sofifa_id``` is the primary key of this dataset
2. ```dob```, ```joined``` are dates, and ```contract_valid_until``` is a four digit integer for a year. 
3. ```overall```, ```potential```, ```pace``` to ```gk_positioning```, ```attacking_crossing``` to 
   ```goalkeeping_reflexes``` , and ```ls``` to ```rb``` are integer scores with range 0-100 (mainly two digits scores)
4. ```international_reputation```, ```weak_foot```, and ```skill_moves``` are integer scores with range 1-5. 
5. ```prefered_foot```, ```real_face``` are binary characters with levels left/right and yes/no respectively. 
6. ```value_eur```, ```wage_eur```, and ```release_clause_eur``` are integer representing money in euros. 
7. ```height_cm``` are three digits integers, and ```weight_kg``` are two to three digits integers. 