class Orbit: #Creating orbit class
    def __init__(self,orbitNumber,startTime,endTime): #Initiation function
        self.orbitNumber = orbitNumber #Creating attributes to store each bit of required data
        self.startTime = startTime
        self.endTime = endTime
        self.calParams = np.empty((3,6,4,)) #Calibration stats are set to nan as default; [axes][range][offset,gain,theta,phi]
        self.calParams[:] = np.nan
    def setF074(self,F074): #Sets the FGM temperature (F074); done like this so that more params can be added later
        self.F074 = F074 #All telems are the averages across an orbit
    def setF034(self,F034): #-12V voltage line
        self.F034 = F034
    def setF047(self,F047): #+5V voltage line
        self.F047 = F047
    def setF048(self,F048): #+12V voltage line
        self.F048 = F048
    def setF055(self,F055): #PSU temp
        self.F055 = F055
    def setJ236(self,J236): #Instrument current
        self.J236 = J236      
          
class OnPeriod: #Object to store all the orbit data beween two switch-on moments; this will include the off period; however, because the data for the off periods will be NaN, it doesn't matter
    def __init__(self,startTime, endTime): #The start of the current on period and the start of the next on period 
        self.startTime = startTime #Defining the start and ends so that the relevant orbits can be searched for
        self.endTime = endTime
        self.periodOrbits = []
