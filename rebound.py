import rebound
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import csv
import matplotlib
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation
import glob

from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from tqdm import tqdm

class Particle:
    """
    Particle Class.

    Parameters
    ----------
    m: scalar
        Mass of the particle in solar masses.
    x,y,z: scalars
        Distances of the particle from the origin in AU.
    vx,vy,vz: scalars
        Initial velocity components of the particle in AU/yr.
    """
    def __init__(self,m,x=0,y=0,z=0,vx=0,vy=0,vz=0):
        self.m = m
        self.x = x
        self.y = y
        self.z = z
        self.a = np.linalg.norm([x, y, z])
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.vi = np.linalg.norm([vx, vy, vz])


class StarSystem:
    G = 39.476926421373

    """
    Star System Class.

    Parameters
    ----------
    stars: list
        List of star objects.
    ast: particle object
        Particle object.
    n_ast: int
        Number of asteroids.
    seed: int
        Seed with which the asteroids initial position
        angles will be randomly generated.
    """

    def __init__(self,stars,ast,n_ast,seed=0):
        self.stars = stars
        self.ast = ast
        self.n_ast = n_ast
        self.seed = seed


    def spawnAsteroid(self):
        global G
        spherical_to_cartesian = lambda rho,phi,theta : [rho*np.sin(phi)*np.cos(theta),
                                                          rho*np.cos(phi)*np.sin(theta),
                                                          rho*np.cos(phi)]
        random_angles = lambda : [2*np.pi*np.random.random_sample(),np.pi*np.random.random_sample()]
        [theta,phi] = random_angles()
        r = spherical_to_cartesian(self.ast.a,phi,theta)
        M = 0
        for star in self.stars:
            M += star.m
        rho = np.random.normal(0,(1/3)*np.sqrt(G*M/self.ast.a),1)
#         rho = np.sqrt(G*M/self.ast.a) * np.random.random_sample()
        while (rho < 0 or rho > np.sqrt(G*M/self.ast.a)):
#             rho = np.sqrt(G*M/self.ast.a) * np.random.random_sample()
            rho = np.random.normal(0,(1/3)*np.sqrt(G*M/self.ast.a),1)
        print(type(r))
        print(type(-rho/np.linalg.norm(r)))
        v_r = (-rho/np.linalg.norm(r))*r
        k = np.sqrt(G*M/self.ast.a - np.linalg.norm(v_r)**2)
        m = 2*np.random.random_sample() - 1
        n = 2*np.random.random_sample() - 1
        v_t = m*np.array([1,0,-r[0]/r[2]]) + n*np.array([0,1,-r[1]/r[2]])
        v_t = (k/np.linalg.norm(v_t))*v_t
        v = v_r + v_t
        return r,v


    def initiateSim(self):
        """
        Creates simulation in Rebound.

        Returns
        ----------
        sim: Rebound simulation object
            Simulation object with parameters of the BinaryStarClass.
        """
        sim = rebound.Simulation()
        sim.units = ("yr","AU","Msun")
        sim.G = 39.476926421373

        for star in self.stars:
            sim.add(m=star.m,
                    x=star.x,y=star.y,z=star.z,
                    vx=star.vx,vy=star.vy,vz=star.vz)

        for i in range(self.n_ast):
            ast = self.spawnAsteroid()
            sim.add(m=self.ast.m,x=ast[0][0],y=ast[0][1],z=ast[0][2],vx=ast[1][0],vy=ast[1][1],vz=ast[1][2])
        return sim


    def integrateSim(self,t_start,t_stop,iterations,sim,dt="reverselog",logbase=np.e):
        """
        Integrates simulation for a set time and iteration count.

        Parameters
        ----------
        t: scalar
            End time of the integration in years
        iterations: int
            Number of times the integration is performed
        sim: Rebound.Simulation() object
            Rebound simulation
        dt: String
            Specifies whether to integrate linearly or logarithmically
        logbase: scalar
            Specifies the base of the log, should the user choose logarithmic integration

        Returns
        ----------
        data: list
            List of dataframes, each containing the t,x,y,z,d,vx,vy,vz,v
            of each star and asteroid for every integration iteration.
        """
        global G
        mass = 0
        for i in range(len(self.stars)):
            mass += self.stars[i].m
        for i in range(self.n_ast):
            mass += self.ast.m
        print(mass)
        data = [np.zeros((iterations,11)) for i in range(sim.N)]

        if dt == "reverselog":
            timespace = t_start+t_stop-np.flipud(np.logspace(start=0,
                                                stop=np.log(t_stop)/np.log(logbase),
                                                num=iterations,
                                                base=logbase))
        elif dt == "linear":
            timespace = np.linspace(start=t_start,stop=t_stop,num=iterations)
        else:
            dt = input("Enter \"reverselog\" or \"linear\": ")
            return self.integrateSim(t_start,t_stop,iterations,sim,dt)

        i = 0
        for time in tqdm(timespace):
            clear_output(wait=True)
            sim.integrate(time)
            for j in range(sim.N):
                data[j][i,0:1] = time
                data[j][i,1:4] = sim.particles[j].xyz
                data[j][i,4:5] = np.linalg.norm(data[j][i,1:4])
                data[j][i,5:8] = np.array(sim.particles[j].vxyz)#*4.74372
                data[j][i,8:9] = np.linalg.norm(data[j][i,5:8])



                d_1 = np.sqrt((data[j][i,1:2] - data[0][i,1:2])**2 + (data[j][i,2:3] - data[0][i,2:3])**2 + (data[j][i,3:4] - data[0][i,4:5])**2)
                d_2 = np.sqrt((data[j][i,1:2] - data[1][i,1:2])**2 + (data[j][i,2:3] - data[1][i,2:3])**2 + (data[j][i,3:4] - data[1][i,4:5])**2)
                v = data[j][i,8:9]
                escapeV = np.sqrt((2 * sim.G * self.stars[0].m / d_1) + (2 * sim.G * self.stars[1].m / d_2))

                rnorm = data[j][i,4:5]
                rdot = data[j][i,5:8]
                r = data[j][i,1:4]
                a = (2*rnorm**-1 - np.dot(rdot,rdot))**-1
                e = np.sqrt(1 - (np.linalg.norm(np.cross(r,rdot)))**2 / a)

#                 if e >= 1:
#                     data[j][i,9:10] = 1
#                 else:
#                     data[j][i,9:10] = 0
                if v > escapeV:
                    data[j][i,9:10] = 1
                else:
                    data[j][i,9:10] = 0
                data[j][i, 10:11] = escapeV
            i += 1

        cols = "t x y z d vx vy vz v e ev".split()
        df_particles = [pd.DataFrame(data=data[i],columns=cols) for i in range(sim.N)]
        return df_particles


    def createSim(self,df_particles):
        sim = rebound.Simulation()
        sim.units = ("yr","AU","Msun")
        sim.G = 39.476926421373
        particle_index = 0
        for df in df_particles:
            if particle_index < len(self.stars):
                sim.add(m = self.stars[particle_index].m,
                        x = df['x'].iloc[-1],
                        y = df['y'].iloc[-1],
                        z = df['z'].iloc[-1],
                        vx = df['vx'].iloc[-1],
                        vy = df['vy'].iloc[-1],
                        vz = df['vz'].iloc[-1])
            else:
                sim.add(m = self.ast.m,
                        x = df['x'].iloc[-1],
                        y = df['y'].iloc[-1],
                        z = df['z'].iloc[-1],
                        vx = df['vx'].iloc[-1],
                        vy = df['vy'].iloc[-1],
                        vz = df['vz'].iloc[-1])
        return sim


    def consecutiveIntegrate(self,sim,t,no_of_sims,iterations_per_sim):
        t_gained = 0
        df_particles = self.integrateSim(sim,t_gained,t_gained+t/no_of_sims,iterations_per_sim,dt="linear")
        t += t/no_of_sims
        while t_gained < t:
            sim = self.createSim(df_particles)
            df = self.integrateSim(t_gained,t_gained+t/no_of_sims,iterations_per_sim,sim,dt="linear")
            for index in range(len(df_particles)):
                pd.concat([df_particles[index],df[index]])
            t_gained += t/no_of_sims
        return df_particles


    def mkpath(self,df_particles):
        """
        Creates path for data on filesystem

        Parameters
        ----------
        df_particles: list
            List of pandas DataFrames, each containing the t,x,y,z,d,vx,vy,vz,d
            of each star and asteroid for every integration iteration.

        Returns
        ----------
        path: String
            Name of the path created relative to ./
        """
        dir_name = ""
        for star in self.stars:
            dir_name += "Star({}M_{}AU)_".format(star.m,star.a)

        dir_name += "Asteroids({}AU_{}V_{}N)".format(self.ast.a,self.ast.vi,self.n_ast)

        t_f = int(df_particles[0]['t'].iloc[-1])
        iterations = df_particles[0].shape[0]
        sub_dir = "{}t_{}iterations".format(t_f,iterations)
        path = "{}/{}".format(dir_name,sub_dir)

        try:
            os.makedirs(path)
            print(dir_name + " created.")
        except FileExistsError:
            print(dir_name + " already exists.")
        finally:
            return path


    def finalPositionsVelocities(self,df_particles,printData=False,saveData=False):
        """
        Returns a dataframe containing the final positions and velocities
        of all the particles.

        Parameters
        ----------
        df_particles: list
            List of pandas DataFrames, each containing the t,x,y,z,d,vx,vy,vz,d
            of each star and asteroid for every integration iteration.
        printData: boolean
            Determines whether to print the data to the console.
        saveData: boolean
            Determines whether to save the data to the hard disk.

        Returns
        ----------
        df: pandas DataFrame
            DataFrame containing the final distances and velocities
            of each star and asteroid.
        """
        global G
        d_f = [particle['d'].iloc[-1] for particle in df_particles]
        v_f = [particle['v'].iloc[-1] for particle in df_particles]
        x_f = [particle['x'].iloc[-1] for particle in df_particles]
        y_f = [particle['y'].iloc[-1] for particle in df_particles]

        star_cols = ["star{}".format(i+1) for i in range(len(self.stars))]
        ast_cols = ["asteroid{}".format(i+1) for i in range(self.n_ast)]
        cols = star_cols + ast_cols

        df = pd.DataFrame(data=np.array([d_f,v_f, x_f, y_f]),columns=cols,index="d v x y".split()).T

        if printData:
            print(df.head())
        if saveData:
            path = self.mkpath(df_particles)
            df.to_csv("{}/finalPositionsVelocities.csv".format(path))
        return df


    def writeData(self,df_particles):
        """
        Creates folder and writes all the data to csv's.

        Parameters
        ----------
        df_particles: list
            List of pandas DataFrames, each containing the t,x,y,z,d,vx,vy,vz,d
            of each star and asteroid for every integration iteration.
        """
        path = self.mkpath(df_particles)
        for i in tqdm(range(len(df_particles))):
            if i < len(self.stars):
                df_particles[i].to_csv("{}/Star{}.csv".format(path,i+1))
            else:
                df_particles[i].to_csv("{}/Asteroid{}.csv".format(path,i+1-len(self.stars)))


    def plotAll(self,df_particles,dim,fname="plot",proj='xy',lim=10,savePlot=True):
        """
        Returns 2D or 3D plot of the given dataframe.

        Parameters
        ----------
        df_particles: list
            List of pandas DataFrames, each containing the t,x,y,z,d,vx,vy,vz,d
            of each star and asteroid for every integration iteration.
        fname: String
            Name of figure.
        dim: int
            2D or 3D (enter 2 or 3)
        savePlot: boolean
            Determines whether the plot is to be saved on the hard disk
        proj: String
            Determines what projection the 2 plot will be

        Returns
        ----------
        fig: matplotlib figure object
            2D or 3D plot of orbit

        """
        fig = plt.figure()
        if (dim!=2 and dim!=3):
            dim = int(input("Dimension must be 2 or 3: "))
            return self.plot(df_particles,fname,dim,proj,savePlot)
        elif (dim == 2):
            ax = fig.add_subplot(111)
            for i in range(len(df_particles)):
                if i < len(self.stars):
                    ax.plot(df_particles[i][proj[0]],
                            df_particles[i][proj[1]],"k")
                else:
                    ax.plot(df_particles[i][proj[0]],
                            df_particles[i][proj[1]])
        else:
            ax = fig.add_subplot(111, projection='3d')
            for i in range(len(df_particles)):
                if i < len(self.stars):
                    ax.plot(df_particles[i]['x'],
                            df_particles[i]['y'],
                            df_particles[i]['z'],"k")
                else:
                    ax.plot(df_particles[i]['x'],
                            df_particles[i]['y'],
                            df_particles[i]['z'])
            ax.set_zlim(-lim,lim)
            ax.set_zlabel("z (AU)")
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_xlabel("{} (AU)".format(proj[0]))
        ax.set_ylabel("{} (AU)".format(proj[1]))
        fname = "{}_{}D".format(fname,dim)
        plt.title(fname)
        path = self.mkpath(df_particles)
        if savePlot:
            plt.savefig("{}/{}.pdf".format(path,fname))
            plt.close(fig)
        return fig


    def plot(self,df_particles,particles=[0],fname="plot",lim=10,savePlot=True):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for particle in particles:
            ax.plot(df_particles[particle]['x'],
                    df_particles[particle]['y'],
                    df_particles[particle]['z'])
        ax.set_zlim(-lim,lim)
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        fname = "{}_{}D".format(fname, "3")
        plt.title(fname)
        path = self.mkpath(df_particles)
        if savePlot:
            plt.savefig("{}/{}.pdf".format(path,fname))
            plt.close(fig)
        return fig


# 100, 1000, 10000 (AU)
# 100 asteroids
# asteroid v_t = sqrt(GM/R), v_r = gaussian from 0 to sqrt(GM/R)
# Run for 10 MYr (1.0e7)
# Send Idan the data tables for AB and proxima, as well as the final figures
# Meeting Idan on Tuesday, 1:00 pm or 2:00 pm





# 2/14
# plots go off of 1 in 10 or 1 in a 100 data points
# plots for all 100 asteroids (with stars)
# functions determining how many asteroids are ejected and how many are consumed by the system using both
# distance parameters and eccentricity parameters
# Consider timestep(linear vs. reverse log vs. new function)
# only run simulation with proxima if starting positions of asteroids = 10,000 AU
# write function to make sure mechanical energy is conserved throughout the system


G = 39.476926421373
stars = [Particle(m=1,x=9,y=10),Particle(m=1,x=-9)] #Particle(m=0.1221,x=8700)
stars[0].vy = np.sqrt(G*stars[1].m/(2*(stars[0].a+stars[1].a)))
stars[1].vy = -np.sqrt(G*stars[0].m/(2*(stars[0].a+stars[1].a)))
ast = Particle(m=0,x=10)
sys = StarSystem(stars,ast,n_ast=1,seed=6)
t_start = 0
t_stop = 100
iterations = 100
sim = sys.initiateSim()
df_particles = sys.integrateSim(t_start,t_stop,iterations,sim,dt="linear")
sys.writeData(df_particles)
df = sys.finalPositionsVelocities(df_particles,printData=True,saveData=True)
sys.plotAll(df_particles,dim=3,lim=25)



# Graph of ejected particles over time
def ejectedOverTime(time=1000):
    x = []
    y = []
    df_length = len(df_particles)
    numStars = len(sys.stars)

    for i in range (len(df_particles[1]['e'])):
        nEjected = 0
        x.append(i)
        for j in range(df_length - numStars):
            if df_particles[j + numStars]['e'][i] == 1:
                nEjected += 1
        y.append(nEjected/(df_length - numStars) * 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0,110)
    ax.set_xlim(0,time)
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage of particles remaining in the system")
    plt.plot(x, y, 'g-')


ejectedOverTime()


# Separate Plotting cell, individual plots


# allows animation to exceed default byte restrictions
matplotlib.rcParams['animation.embed_limit'] = 2**128


def plotIn(particles,lim=50,savePlot=True,x=5):
    df_particles = []
    files = glob.glob(particles + "*.csv")
    for i in range(len(files)):
        if files[i].endswith(".csv"):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            data = {'x': [], 'y': [], 'z': []}
            with open(files[i]) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                oneinx = 0
                for row in tqdm(csv_reader):
                    oneinx += 1
                    if oneinx%x == 0:
                        if line_count > 0:
                            data['x'].append(float(row[2]))
                            data['y'].append(float(row[3]))
                            data['z'].append(float(row[4]))
                        line_count += 1
            df_particles.append(pd.DataFrame(data=data))
            ax.plot(df_particles[i]['x'],
                    df_particles[i]['y'],
                    df_particles[i]['z'])
            ax.set_zlim(-lim,lim)
            ax.set_xlim(-lim,lim)
            ax.set_ylim(-lim,lim)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            name = files[i].replace(particles, '')
            fname = "{}_{}".format("plot", name)
            plt.title(fname)
            if savePlot:
                plt.savefig("{}_{}.pdf".format("plot", name))
                plt.close(fig)
            return fig


particles = "Star(1M_9.0AU)_Star(1M_9.0AU)_Asteroids(100.0AU_0.0V_2N)/10000t_1000iterations/"



# Statistical analysis of data

files = []
finalPositionsVelocities = "Star(1M_9.0AU)_Star(1M_9.0AU)_Asteroids(10.0AU_0.0V_20N)/100t_100iterations/finalPositionsVelocities.csv"

def statistics(df):
        """
        Returns the mean and standard deviation of the final positions and velocities.

        Parameters
        ----------
        df: Pandas DataFrame object
            Dataframe containing the final velocities and positions of each particle.

        Returns
        ----------
        df: Pandas DataFrame object
            Dataframe containing the mean and std of the velocities and positions of the particles
        """

        data = pd.DataFrame(data=np.array([[np.mean(df['v']),np.mean(df['d'])],
                                           [np.std(df['v']),np.std(df['d'])]]),
                            columns='v d'.split(),
                            index='mu sigma'.split())
        return data


def num_ejected(finalNumbers, maxDist):
    num = 0
    with open(finalNumbers) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if 'asteroid' in row[0]:
                if float(row[1]) > maxDist:
                    num += 1
    print("Number of asteroids farther than " + str(maxDist) + " AU: " + str(num))



def histogram(df):
        """
        Returns a histogram of the final positions and velocities.

        Parameters
        ----------
        df: Pandas DataFrame object
            Dataframe containing the final velocities and positions of each particle.

        Returns
        ----------
        fig: Matplotlib figure object
            Figure with subplots of the histograms.
        """
        fig,axs = plt.subplots(1,2)
        axs[0].hist(df['d'])
        axs[1].hist(df['v'])
        axs[0].set_title('d')
        axs[1].set_title('v')
        plt.show()
        return fig


def scatter(finalNumbers):
    x = []
    y = []
    with open(finalNumbers) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if 'asteroid' in row[0]:
                x.append(row[2])
                y.append(row[3])
    plt.scatter(x, y, alpha=0.5)

# joint plot with x,y positions of asteroids
# plots with final positions relative to starting velocities (radial, tangential, and total) (hexbin)
# plots with final positions relative to starting positions (10 AU, 100 AU, 1000 AU)
# add to statistics function


histogram(df)
# statistics(df)
num_ejected(finalPositionsVelocities, 100)

# Separate plotting cell, plot one in x data points and produced individual plots for all particles

def plotLess(particles=[],fname="plot",proj='xy',lim=100,savePlot=True,x=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    df_particles = []
    for i in range(len(particles)):
        data = {'x': [], 'y': [], 'z': []}
        with open(particles[i]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            oneinx = 0
            for row in csv_reader:
                oneinx += 1
                if oneinx%x == 0:
                    if line_count > 0:
                        data['x'].append(float(row[2]))
                        data['y'].append(float(row[3]))
                        data['z'].append(float(row[4]))
                    line_count += 1
        df_particles.append(pd.DataFrame(data=data))
    for i in range(len(particles)):
        ax.plot(df_particles[i]['x'],
                df_particles[i]['y'],
                df_particles[i]['z'])
    ax.set_zlim(-lim,lim)
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    fname = "{}_{}D".format(fname, "3")
    plt.title(fname)
    if savePlot:
        plt.savefig("{}.pdf".format(fname))
        plt.close(fig)
    return fig


particles = [
             "Star(1M_9.0AU)_Star(1M_9.0AU)_Asteroids(100.0AU_0.0V_2N)/10000t_1000iterations/Asteroid2.csv",
             "Star(1M_9.0AU)_Star(1M_9.0AU)_Asteroids(100.0AU_0.0V_2N)/10000t_1000iterations/Asteroid1.csv"
            ]
plotLess(particles)


# 3 dimensional animation of multiple paths (data from csv files)

# allows animation to exceed default byte restrictions
matplotlib.rcParams['animation.embed_limit'] = 2**128

# initialized data arrays containing all data in csv files
xdata = []
ydata = []
zdata = []

# intializes data arrays to be plotted differently in each frame
x_new = []
y_new = []
z_new = []

# initialized arrays for line line objects
lines = []

# insert csv files to be plotted
files = [#'Asteroid3.csv', 'Asteroid2.csv',
         'Star2.csv', 'Star1.csv', 'Star3.csv',
         #'Asteroid4 (1).csv', 'Asteroid5.csv',
         #'Asteroid6.csv', 'Asteroid7.csv', 'Asteroid8.csv',
         #'Asteroid10.csv', 'Asteroid6.csv'
        ]

# different colors for each line
colors = ['g-', 'b-', 'm-', 'c-', 'y-', 'k']
len_colors = len(colors)
files_length = len(files)

# initializes plot, first line, and first axis
fig = plt.figure()
ax = p3.Axes3D(fig)
l, = ax.plot([],[],[], 'r-', animated=True)
lines.append(l)

# creates new empty line for each file in files
for i in range(files_length):
    lines.append(ax.plot([],[],[], colors[i % len_colors], animated=True)[0])

# makes the data arrays 2 dimensional, adds an empty array for each file in files
for i in range(files_length):
    xdata.append([])
    ydata.append([])
    zdata.append([])
    x_new.append([])
    y_new.append([])
    z_new.append([])

# reads the csv files and writes the data in the data arrays
for i in range(files_length):
    with open(files[i]) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                xdata[i].append(float(row[1]))
                ydata[i].append(float(row[2]))
                zdata[i].append(float(row[3]))
            line_count += 1

# determines how many frames there should be in the animation (one frame for each datapoint)
length = len(xdata[0])


# initialization function for animation, sets all axes lengths
def init():
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    return tuple(lines)


# update function, repeatedly called by the animation object, updates the displayed data arrays and changes
# line objects accordingly
def animate(i):
    for j in range(files_length):
        x_new[j].append(xdata[j][i])
        y_new[j].append(ydata[j][i])
        z_new[j].append(zdata[j][i])

    for j in range(files_length):
        lines[j].set_data(x_new[j], y_new[j])
        lines[j].set_3d_properties(z_new[j])
    return tuple(lines)

# creates animation object using the created figure, initializes it and updates frames
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=length,
                                        init_func=init, blit=True)

# displays animation
from IPython.display import HTML
plt.close()
HTML(ani.to_jshtml())
