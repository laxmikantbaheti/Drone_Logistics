- All instances are derived from the CVRPLIB publicly available at http://vrp.atd-lab.inf.puc-rio.br/index.php/en/
	
- We mainly use instances from set A and set B (Augerat1995) and consider the first 20, 30, 40, or 50 vertices, respectively.

- We add an additional number at the end of the instance name to specifiy the number of vertices considered,
	e.g. A-n38-k5-20.vrp => 20 vertices are considered. 
	
- All customers with a demand greater than CAPACITY / 5 are truck-only customers.

- We compute the travel times/costs for truck and drone in the following way:
	- Truck (Manhattan Distance):
	
		double dist = (double)abs(v_xCoord[i] - v_xCoord[j]) + abs(v_yCoord[i] - v_yCoord[j]);
		m_Trucktimes(i, j) = (int)floor(Rounding * dist);
	
	- Drone (Euclidean Distance):
	
		double dist2 = sqrt((double)(v_xCoord[i] - v_xCoord[j]) * (v_xCoord[i] - v_xCoord[j]) + (v_yCoord[i] - v_yCoord[j]) * (v_yCoord[i] - v_yCoord[j])) / beta;
		
		m_Dronetimes(i, j) = (int)(floor(Rounding * dist2));

	- The parameter Rounding is set to 10.0 and beta represetns the drone's speed, e.g., 1.0, 3.0 and 5.0.