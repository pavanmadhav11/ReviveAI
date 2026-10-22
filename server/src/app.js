import express from 'express';
import cors from 'cors';
import multer from 'multer';
import axios from 'axios';
import FormData from 'form-data';
import Restoration from './models/Restoration.js';
import mongoose from 'mongoose';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {existsSync} from 'node:fs';

const app = express();
app.use(cors({origin:process.env.CLIENT_ORIGIN || 'http://localhost:5173'}));
app.use(express.json({limit:'1mb'}));
const upload = multer({storage:multer.memoryStorage(),limits:{fileSize:(Number(process.env.MAX_UPLOAD_MB)||12)*1024*1024},fileFilter:(_req,file,cb)=>{
  const ok=['image/jpeg','image/png','image/webp'].includes(file.mimetype); cb(ok?null:new Error('Only JPG, PNG, and WEBP images are supported.'),ok);
}});
const dip = process.env.DIP_URL || 'http://127.0.0.1:8000';
const requireImage = (req,res,next)=>upload.single('image')(req,res,err=>{
  if(err) return res.status(400).json({error:err.message});
  if(!req.file) return res.status(400).json({error:'An image file is required.'}); next();
});

app.get('/api/health',async (_req,res)=>{
  let dipStatus='unavailable'; try { await axios.get(`${dip}/health`,{timeout:1500}); dipStatus='ok'; } catch {}
  res.json({status:'ok',dipService:dipStatus});
});
app.post('/api/restoration/restore',requireImage,async(req,res,next)=>{
  try {
    const form=new FormData(); form.append('image',req.file.buffer,{filename:req.file.originalname,contentType:req.file.mimetype});
    const options=typeof req.body.options==='string'?req.body.options:JSON.stringify(req.body.options||{}); form.append('options',options);
    const {data}=await axios.post(`${dip}/process`,form,{headers:form.getHeaders(),maxBodyLength:25*1024*1024,timeout:120000});
    if(mongoose.connection.readyState===1) try { await Restoration.create({originalFilename:req.file.originalname,mode:JSON.parse(options).mode||'full',parameters:JSON.parse(options),durationMs:data.durationMs,dimensions:data.dimensions}); } catch {}
    res.json(data);
  } catch(e) { next(e); }
});
app.post('/api/restoration/evaluate',upload.fields([{name:'reference',maxCount:1},{name:'restored',maxCount:1}]),async(req,res,next)=>{
  try { const ref=req.files?.reference?.[0], out=req.files?.restored?.[0]; if(!ref||!out) return res.status(400).json({error:'Reference and restored images are required.'});
    const f=new FormData(); f.append('reference',ref.buffer,{filename:ref.originalname,contentType:ref.mimetype}); f.append('restored',out.buffer,{filename:out.originalname,contentType:out.mimetype});
    const {data}=await axios.post(`${dip}/evaluate`,f,{headers:f.getHeaders(),timeout:30000}); res.json(data);
  } catch(e){ next(e); }
});
app.get('/api/restoration/history',async(_req,res)=>{try{res.json({items:await Restoration.find().sort({createdAt:-1}).limit(20).lean(),available:true});}catch{res.json({items:[],available:false});}});
// Serve the already-built client from the same local origin for a simple one-URL run.
const clientDist=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../client/dist');
if(existsSync(clientDist)){
  app.use(express.static(clientDist));
  app.get('*',(_req,res)=>res.sendFile(path.join(clientDist,'index.html')));
}
app.use((err,_req,res,_next)=>{const status=err.response?.status||500;res.status(status).json({error:err.response?.data?.detail||err.message||'Restoration request failed.'});});
export default app;
