import 'dotenv/config';
import mongoose from 'mongoose';
import app from './app.js';
const port=process.env.PORT||5000;
if(process.env.MONGODB_URI) mongoose.connect(process.env.MONGODB_URI).catch(e=>console.warn('MongoDB unavailable; history disabled:',e.message));
const host=process.env.HOST||'127.0.0.1';
app.listen(port,host,()=>console.log(`ReviveAI API listening on http://${host}:${port}`));
