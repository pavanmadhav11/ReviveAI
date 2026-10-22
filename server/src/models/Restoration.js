import mongoose from 'mongoose';
const schema = new mongoose.Schema({originalFilename:String, mode:String, parameters:Object, durationMs:Number, dimensions:Object, createdAt:{type:Date,default:Date.now}});
export default mongoose.model('Restoration', schema);
