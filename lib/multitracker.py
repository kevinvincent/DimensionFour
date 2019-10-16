
class Multitracker():
   def __init__(self, maxDisappeared=10):
      self.trackerObjs = []

   def add(self, tracker, id):
      self.trackerObjs.append({
         "tracker": tracker,
         "id": id
      })

   def update(self, frame):
      toRemove = []
      for (i, trackerObj) in enumerate(self.trackerObjs):
         status, bbox = trackerObj["tracker"].update(frame)
         if status:
            trackerObj["status"] = status
            trackerObj["bbox"] = bbox
         else:
            toRemove.append(i)

      for index in toRemove:
         del self.trackerObjs[index]
         
      return self.trackerObjs
      
         
