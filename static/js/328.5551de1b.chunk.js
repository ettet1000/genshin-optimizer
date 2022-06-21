"use strict";(self.webpackChunkgenshin_optimizer=self.webpackChunkgenshin_optimizer||[]).push([[328],{31038:function(e,t,n){n.d(t,{Z:function(){return I}});var r=n(1413),i=n(93433),a=n(45987),l=n(53174),o=n(54483),c=n(63204),s=n(14361),u=n(13967),d=n(47047),f=n(14868),h=n(20890),p=n(68870),v=n(72791),x=n(22020),Z=n(947),y=n(56928),m=n(42320),b=n(24351),g=n(50921),j=n(46956),k=n(610),w=n(68244),C=n(2380),S=n(80184),E=["value","onChange","defaultText","defaultIcon","placeholderText","labelText","showDefault","showInventory","showEquipped","filter","disable"];function I(e){var t=e.value,n=e.onChange,I=e.defaultText,M=void 0===I?"":I,_=e.defaultIcon,F=void 0===_?"":_,A=e.placeholderText,K=void 0===A?"":A,P=e.labelText,R=void 0===P?"":P,O=e.showDefault,T=void 0!==O&&O,G=e.showInventory,V=void 0!==G&&G,D=e.showEquipped,N=void 0!==D&&D,q=e.filter,z=void 0===q?function(){return!0}:q,B=e.disable,L=void 0===B?function(){return!1}:B,W=(0,a.Z)(e,E),H=(0,x.$)(["ui","artifact"].concat((0,i.Z)(b.IV.map((function(e){return"char_".concat(e,"_gen")}))))).t,$=(0,u.Z)(),U=(0,v.useContext)(y.t).database,J=(0,m.Z)(Z.Z.getAll,[]),Q=(0,v.useMemo)((function(){return J&&(0,g.zU)(U,J)}),[U,J]),X=U._getCharKeys().filter((function(e){return(null===J||void 0===J?void 0:J[e])&&z(J[e],e)})).sort(),Y=(0,v.useCallback)((function(e){switch(e){case"Equipped":return H("artifact:filterLocation.currentlyEquipped");case"Inventory":return H("artifact:filterLocation.inventory");case"":return M;default:return H("char_".concat(e,"_gen:name"))}}),[M,H]),ee=(0,v.useCallback)((function(e){var t;switch(e){case"Equipped":return(0,S.jsx)(o.G,{icon:l.caW});case"Inventory":return(0,S.jsx)(c.Z,{});case"":return F;default:return(0,S.jsx)(C.Z,{src:null===(t=J[e])||void 0===t?void 0:t.thumbImgSide,sx:{pr:1}})}}),[F,J]),te=(0,v.useMemo)((function(){return Q&&function(e,t,n,r,a,l){if(!t)return[];var o=[];r&&o.push({value:"",label:n("")});a&&o.push({value:"Inventory",label:n("Inventory")});l&&o.push({value:"Equipped",label:n("Equipped")});var c=e.filter((0,j.C)({element:(0,i.Z)(b.N),weaponType:(0,i.Z)(b.yd),favorite:"yes",name:""},t)).map((function(e){return{value:e,label:n(e)}})),s=e.filter((0,j.C)({element:(0,i.Z)(b.N),weaponType:(0,i.Z)(b.yd),favorite:"no",name:""},t)).map((function(e){return{value:e,label:n(e)}}));return o.concat(c).concat(s)}(X,Q,Y,T,V,N)}),[Q,X,T,V,N,Y]);return J&&te?(0,S.jsx)(f.Z,(0,r.Z)({autoHighlight:!0,options:te,clearIcon:t?void 0:"",getOptionLabel:function(e){return e.label},onChange:function(e,t,r){return("change"!==e.type||"clear"!==r)&&n(t?t.value:"")},isOptionEqualToValue:function(e,t){return e.value===t.value},getOptionDisabled:function(e){return L(e.value)},value:{value:t,label:Y(t)},renderInput:function(e){return(0,S.jsx)(w.Z,(0,r.Z)((0,r.Z)({},e),{},{label:R,placeholder:K,startAdornment:ee(t),hasValue:!!t}))},renderOption:function(e,n){var r,i="Equipped"!==n.value&&"Inventory"!==n.value&&""!==n.value&&(null===(r=U._getChar(n.value))||void 0===r?void 0:r.favorite);return(0,S.jsx)(k.Z,{value:n.value?n.value:"default",image:ee(n.value),text:(0,S.jsx)(v.Suspense,{fallback:(0,S.jsx)(d.Z,{variant:"text",width:100}),children:(0,S.jsx)(h.Z,{variant:"inherit",noWrap:!0,children:Y(n.value)})}),theme:$,isSelected:t===n.value,addlElement:(0,S.jsxs)(S.Fragment,{children:[i&&(0,S.jsx)(p.Z,{display:"flex",flexGrow:1}),i&&(0,S.jsx)(s.Z,{sx:{ml:1,mr:-.5}})]}),props:e},n.value?n.value:"default")}},W)):(0,S.jsx)(d.Z,{height:50})}},40020:function(e,t,n){n.d(t,{Z:function(){return s}});var r=n(63204),i=n(20890),a=n(22020),l=n(947),o=n(42320),c=n(80184);function s(e){var t=e.location,n=(0,a.$)("ui").t,s=(0,o.Z)(l.Z.get(null!==t&&void 0!==t?t:""),[t]);return(0,c.jsx)(i.Z,{component:"span",children:null!==s&&void 0!==s&&s.name?s.nameWIthIcon:(0,c.jsxs)("span",{children:[(0,c.jsx)(r.Z,{sx:{verticalAlign:"text-bottom"}})," ",n("inventory")]})})}},2380:function(e,t,n){var r=(0,n(93457).Z)("img")((function(e){var t=e.theme;return{display:"inline-block",width:"auto",height:"2.3em",lineHeight:1,verticalAlign:"text-bottom",marginTop:t.spacing(-3),marginLeft:t.spacing(-1.25),marginRight:t.spacing(-1),marginBottom:t.spacing(-1)}}));t.Z=r},10600:function(e,t,n){n.d(t,{CC:function(){return v},ZP:function(){return x}});var r=n(29439),i=n(1413),a=n(45987),l=n(66934),o=n(4834),c=n(24518),s=n(72791),u=n(80184),d=["children","disableRipple","disableFocusRipple","disableTouchRipple"],f=["value","onChange","disabled","float"],h=(0,l.ZP)(o.ZP)((function(e){var t=e.theme;return{backgroundColor:t.palette.primary.main,transition:"all 0.5s ease","&:hover":{backgroundColor:t.palette.primary.dark},"&.Mui-focused":{backgroundColor:t.palette.primary.dark},"&.Mui-disabled":{backgroundColor:t.palette.primary.dark}}})),p=(0,l.ZP)(c.Z)((function(e){return{backgroundColor:e.theme.palette.primary.main,padding:0,overflow:"hidden",div:{width:"100%",height:"100%"}}}));function v(e){var t=e.children,n=(e.disableRipple,e.disableFocusRipple,e.disableTouchRipple,(0,a.Z)(e,d));return(0,u.jsx)(p,(0,i.Z)((0,i.Z)({disableRipple:!0,disableFocusRipple:!0,disableTouchRipple:!0},n),{},{children:t}))}function x(e){var t=e.value,n=void 0===t?0:t,l=e.onChange,o=e.disabled,c=void 0!==o&&o,d=e.float,p=void 0!==d&&d,v=(0,a.Z)(e,f),x=(0,s.useState)(n),Z=(0,r.Z)(x,2),y=Z[0],m=Z[1],b=(0,s.useState)(!1),g=(0,r.Z)(b,2),j=g[0],k=g[1],w=(0,s.useMemo)((function(){return p?parseFloat:parseInt}),[p]),C=(0,s.useCallback)((function(){l(y),k(!1)}),[l,y,k]),S=(0,s.useCallback)((function(){k(!0)}),[k]);(0,s.useEffect)((function(){return m(n)}),[n,m]);var E=(0,s.useCallback)((function(e){return m(w(e.target.value)||0)}),[m,w]),I=(0,s.useCallback)((function(e){return"Enter"===e.key&&C()}),[C]);return(0,u.jsx)(h,(0,i.Z)({value:j&&!y?"":y,"aria-label":"custom-input",type:"number",inputProps:{step:p?.1:1},onChange:E,onBlur:C,onFocus:S,disabled:c,onKeyDown:I},v))}},55221:function(e,t,n){var r=n(1413),i=n(45987),a=n(53174),l=n(54483),o=n(68870),c=n(96106),s=n(80184),u=["className"];t.Z=function(e){var t=e.className,n=(0,i.Z)(e,u);return(0,s.jsx)(c.Z,(0,r.Z)((0,r.Z)({placement:"top"},n),{},{className:t,children:(0,s.jsx)(o.Z,{component:"span",sx:{cursor:"help"},children:(0,s.jsx)(l.G,{icon:a.sqG})})}))}},610:function(e,t,n){n.d(t,{Z:function(){return c}});var r=n(1413),i=n(23786),a=n(57064),l=n(49900),o=n(80184);function c(e){var t=e.value,n=e.image,c=void 0===n?"":n,s=e.text,u=e.theme,d=e.isSelected,f=e.addlElement,h=e.props;return(0,o.jsxs)(i.Z,(0,r.Z)((0,r.Z)({value:t},h),{},{children:[(0,o.jsx)(a.Z,{children:c}),(0,o.jsx)(l.Z,{primaryTypographyProps:{style:{fontWeight:d?u.typography.fontWeightMedium:u.typography.fontWeightRegular}},children:s}),f&&f]}),t)}},68244:function(e,t,n){n.d(t,{Z:function(){return s}});var r=n(1413),i=n(45987),a=n(13967),l=n(58165),o=n(80184),c=["hasValue","startAdornment","flattenCorners","InputProps","sx"];function s(e){var t=e.hasValue,n=e.startAdornment,s=e.flattenCorners,u=void 0!==s&&s,d=e.InputProps,f=e.sx,h=(0,i.Z)(e,c),p=(0,a.Z)();return(0,o.jsx)(l.Z,(0,r.Z)((0,r.Z)({},h),{},{variant:"filled",color:t?"success":"primary",hiddenLabel:!h.label,type:"search",InputProps:(0,r.Z)((0,r.Z)({},d),{},{startAdornment:n}),InputLabelProps:{style:{color:p.palette.text.primary}},sx:(0,r.Z)((0,r.Z)({},f),{},{"& .MuiFilledInput-root":{backgroundColor:t?p.palette.success.main:p.palette.primary.main,borderRadius:u?0:1,paddingTop:h.label?void 0:0,paddingBottom:0},"& .MuiFilledInput-root.Mui-focused":{backgroundColor:t?p.palette.success.light:p.palette.primary.light},"& .MuiFilledInput-root:hover":{backgroundColor:t?p.palette.success.dark:p.palette.primary.dark},"& .MuiFilledInput-root:before":{border:"none"},"& .MuiFilledInput-root.Mui-disabled:before":{border:"none"},"& .MuiFilledInput-root:after":{border:"none"},"& .MuiFilledInput-root:hover:not(.Mui-disabled):before":{border:"none"},"& input[type=search]::-ms-clear":{display:"none",width:0,height:0},"& input[type=search]::-ms-reveal":{display:"none",width:0,height:0},"& input[type=search]::-webkit-search-decoration":{display:"none"},"& input[type=search]::-webkit-search-cancel-button":{display:"none"},"& input[type=search]::-webkit-search-results-button":{display:"none"},"& input[type=search]::-webkit-search-results-decoration":{display:"none"}})}))}},14525:function(e,t,n){n.d(t,{Z:function(){return M},b:function(){return _}});var r,i,a=n(29439),l=n(30168),o=n(4942),c=n(1413),s=n(53174),u=n(54483),d=n(39504),f=n(20890),h=n(2199),p=n(23786),v=n(24518),x=n(52791),Z=n(72791),y=n(22020),m=n(17618),b=n(2693),g=n(26138),j=n(79406),k=n(9274),w=n(71310),C=n(10600),S=n(33890),E=n(55221),I=n(80184);function M(e){var t=e.disabled,n=void 0!==t&&t,s=(0,y.$)("page_character").t,u=(0,Z.useContext)(m.K).character.key,h=(0,Z.useContext)(b.R).data,p=(0,k.ZP)(u),v=p.buildSetting.statFilters,j=p.buildSettingDispatch,C=(0,Z.useCallback)((function(e){return j({statFilters:e})}),[j]),S=(0,Z.useMemo)((function(){var e=["atk","hp","def","eleMas","critRate_","critDMG_","heal_","enerRech_"];"catalyst"!==h.get(g.ri.weaponType).value&&e.push("physical_dmg_");var t=h.get(g.ri.charEle).value;return e.push("".concat(t,"_dmg_")),e}),[h]),M=(0,Z.useMemo)((function(){return S.filter((function(e){return!Object.keys(v).some((function(t){return t===e}))}))}),[S,v]),F=(0,Z.useCallback)((function(e,t){return C((0,c.Z)((0,c.Z)({},v),{},(0,o.Z)({},e,t)))}),[v,C]);return(0,I.jsxs)(x.Z,{children:[(0,I.jsx)(w.Z,{children:(0,I.jsxs)(d.Z,{sx:{display:"flex",gap:1,justifyContent:"space-between"},children:[(0,I.jsx)(f.Z,{children:s(r||(r=(0,l.Z)(["tabOptimize.constraintFilter.title"])))}),(0,I.jsx)(E.Z,{title:(0,I.jsx)(f.Z,{children:s(i||(i=(0,l.Z)(["tabOptimize.constraintFilter.tooltip"])))})})]})}),(0,I.jsxs)(x.Z,{display:"flex",flexDirection:"column",gap:.5,children:[Object.entries(v).map((function(e){var t=(0,a.Z)(e,2),r=t[0],i=t[1];return(0,I.jsx)(_,{statKey:r,statKeys:M,setFilter:F,disabled:n,value:i,close:function(){delete v[r],C((0,c.Z)({},v))}},r)})),(0,I.jsx)(_,{statKeys:M,setFilter:F,disabled:n})]})]})}function _(e){var t=e.statKey,n=e.statKeys,r=void 0===n?[]:n,i=e.value,a=void 0===i?0:i,l=e.close,o=e.setFilter,c=e.disabled,d=void 0!==c&&c,f="%"===j.ZP.unit(t),x=(0,Z.useCallback)((function(e){return t&&o(t,e)}),[o,t]);return(0,I.jsxs)(h.Z,{sx:{width:"100%"},children:[(0,I.jsx)(S.Z,{title:t?(0,I.jsx)(j._J,{statKey:t}):"New Stat",disabled:d,color:t?"success":"secondary",children:r.map((function(e){return(0,I.jsx)(p.Z,{onClick:function(){null===l||void 0===l||l(),o(e,a)},children:(0,I.jsx)(j._J,{statKey:e})},e)}))}),(0,I.jsx)(C.CC,{sx:{flexBasis:30,flexGrow:1},children:(0,I.jsx)(C.ZP,{disabled:!t||d,float:f,value:a,placeholder:"Stat Value",onChange:x,sx:{px:2},inputProps:{sx:{textAlign:"right"}},endAdornment:j.ZP.unit(t)})}),!!l&&(0,I.jsx)(v.Z,{color:"error",onClick:l,disabled:d,children:(0,I.jsx)(u.G,{icon:s.I7k})})]})}},20323:function(e,t,n){function r(){return{tcMode:!1}}n.d(t,{c:function(){return r}})},72838:function(e,t,n){n.d(t,{N:function(){return Q},Z:function(){return U}});var r,i,a,l,o=n(30168),c=n(29439),s=n(53174),u=n(54483),d=n(40117),f=n(62002),h=n(63204),p=n(66647),v=n(68870),x=n(47047),Z=n(20890),y=n(13400),m=n(81918),b=n(39504),g=n(2199),j=n(40165),k=n(24518),w=n(72791),C=n(22020),S=n(95614),E=n(71310),I=n(31038),M=n(40020),_=n(91702),F=n(20005),A=n(55221),K=n(25617),P=n(10157),R=n(75545),O=n(19272),T=n(31148),G=n(56928),V=n(79406),D=n(63372),N=n(42320),q=n(50765),z=n(60393),B=n(46797),L=n(44217),W=n(80184),H=(0,w.lazy)((function(){return Promise.all([n.e(788),n.e(213)]).then(n.bind(n,66585))})),$=new Set(q._);function U(e){var t,n,q,z,U=e.artifactId,Q=e.artifactObj,X=e.onClick,Y=e.onDelete,ee=e.mainStatAssumptionLevel,te=void 0===ee?0:ee,ne=e.effFilter,re=void 0===ne?$:ne,ie=e.probabilityFilter,ae=e.disableEditSetSlot,le=void 0!==ae&&ae,oe=e.editor,ce=void 0!==oe&&oe,se=e.canExclude,ue=void 0!==se&&se,de=e.canEquip,fe=void 0!==de&&de,he=e.extraButtons,pe=(0,C.$)(["artifact","ui"]).t,ve=(0,w.useContext)(G.t).database,xe=(0,D.Z)(U),Ze=(0,N.Z)(T.y.get(null===(t=null!==Q&&void 0!==Q?Q:xe)||void 0===t?void 0:t.setKey),[Q,xe]),ye=!Q,me=(0,w.useState)(!1),be=(0,c.Z)(me,2),ge=be[0],je=be[1],ke=(0,w.useCallback)((function(){return je(!1)}),[je]),we=(0,w.useCallback)((function(){return ye&&je(!0)}),[ye,je]),Ce=(0,w.useCallback)((function(e){return(0,W.jsx)(p.Z,{onClick:function(){return U&&(null===X||void 0===X?void 0:X(U))},sx:{flexGrow:1,display:"flex",flexDirection:"column"},children:e})}),[X,U]),Se=(0,w.useCallback)((function(e){return(0,W.jsx)(v.Z,{sx:{flexGrow:1,display:"flex",flexDirection:"column"},children:e})}),[]),Ee=null!==Q&&void 0!==Q?Q:xe;if(!Ee)return null;var Ie=Ee.id,Me=Ee.lock,_e=Ee.slotKey,Fe=Ee.rarity,Ae=Ee.level,Ke=Ee.mainStatKey,Pe=Ee.substats,Re=Ee.exclude,Oe=Ee.location,Te=void 0===Oe?"":Oe,Ge=Math.max(Math.min(te,4*Fe),Ae),Ve=V.ZP.unit(Ke),De="roll"+(Math.floor(Math.max(Ae,0)/4)+1),Ne=O.Z.getArtifactEfficiency(Ee,re),qe=Ne.currentEfficiency,ze=Ne.maxEfficiency,Be=0!==ze,Le=null===Ze||void 0===Ze?void 0:Ze.getSlotName(_e),We=null===Ze||void 0===Ze?void 0:Ze.getSlotDesc(_e),He=We&&(0,W.jsx)(A.Z,{title:(0,W.jsxs)(v.Z,{children:[(0,W.jsx)(w.Suspense,{fallback:(0,W.jsx)(x.Z,{variant:"text",width:100}),children:(0,W.jsx)(Z.Z,{variant:"h6",children:Le})}),(0,W.jsx)(Z.Z,{children:We})]})}),$e=null===Ze||void 0===Ze?void 0:Ze.setEffects,Ue=Ze&&$e&&(0,W.jsx)(A.Z,{title:(0,W.jsx)("span",{children:Object.keys($e).map((function(e){return(0,W.jsxs)("span",{children:[(0,W.jsx)(Z.Z,{variant:"h6",children:(0,W.jsx)(K.Z,{color:"success",children:pe("artifact:setEffectNum",{setNum:e})})}),(0,W.jsx)(Z.Z,{children:Ze.setEffectDesc(e)})]},e)}))})});return(0,W.jsxs)(w.Suspense,{fallback:(0,W.jsx)(x.Z,{variant:"rectangular",sx:{width:"100%",height:"100%",minHeight:350}}),children:[ce&&(0,W.jsx)(w.Suspense,{fallback:!1,children:(0,W.jsx)(H,{artifactIdToEdit:ge?U:"",cancelEdit:ke,disableEditSetSlot:le})}),(0,W.jsxs)(E.Z,{sx:{height:"100%",display:"flex",flexDirection:"column"},children:[(0,W.jsxs)(F.Z,{condition:!!X,wrapper:Ce,falseWrapper:Se,children:[(0,W.jsxs)(v.Z,{className:"grad-".concat(Fe,"star"),sx:{position:"relative",width:"100%"},children:[!X&&(0,W.jsx)(y.Z,{color:"primary",disabled:!ye,onClick:function(){return ve.updateArt({lock:!Me},Ie)},sx:{position:"absolute",right:0,bottom:0,zIndex:2},children:Me?(0,W.jsx)(d.Z,{}):(0,W.jsx)(f.Z,{})}),(0,W.jsxs)(v.Z,{sx:{pt:2,px:2,position:"relative",zIndex:1},children:[(0,W.jsxs)(v.Z,{component:"div",sx:{display:"flex",alignItems:"center",gap:1,mb:1},children:[(0,W.jsx)(m.Z,{size:"small",label:(0,W.jsx)("strong",{children:" +".concat(Ae)}),color:De}),!Le&&(0,W.jsx)(x.Z,{variant:"text",width:100}),Le&&(0,W.jsx)(Z.Z,{noWrap:!0,sx:{textAlign:"center",backgroundColor:"rgba(100,100,100,0.35)",borderRadius:"1em",px:1},children:(0,W.jsx)("strong",{children:Le})}),!He&&(0,W.jsx)(x.Z,{width:10}),He]}),(0,W.jsx)(Z.Z,{color:"text.secondary",variant:"body2",children:(0,W.jsx)(S.ZP,{slotKey:_e})}),(0,W.jsx)(Z.Z,{variant:"h6",color:"".concat(V.ZP.getVariant(Ke),".main"),children:(0,W.jsxs)("span",{children:[R.Z[Ke]," ",V.ZP.get(Ke)]})}),(0,W.jsx)(Z.Z,{variant:"h5",children:(0,W.jsx)("strong",{children:(0,W.jsxs)(_.Z,{color:Ge!==Ae?"warning":void 0,children:[(0,V.qs)(null!==(n=O.Z.mainStatValue(Ke,Fe,Ge))&&void 0!==n?n:0,V.ZP.unit(Ke)),Ve]})})}),(0,W.jsx)(P.t,{stars:Fe,colored:!0})]}),(0,W.jsx)(v.Z,{sx:{height:"100%",position:"absolute",right:0,top:0},children:(0,W.jsx)(v.Z,{component:"img",src:null!==(q=null===Ze||void 0===Ze?void 0:Ze.slotIcons[_e])&&void 0!==q?q:"",width:"auto",height:"100%",sx:{float:"right"}})})]}),(0,W.jsxs)(b.Z,{sx:{flexGrow:1,display:"flex",flexDirection:"column",pt:1,pb:0,width:"100%"},children:[Pe.map((function(e){return(0,W.jsx)(J,{stat:e,effFilter:re,rarity:Fe},e.key)})),(0,W.jsxs)(v.Z,{sx:{display:"flex",my:1},children:[(0,W.jsx)(Z.Z,{color:"text.secondary",component:"span",variant:"caption",sx:{flexGrow:1},children:pe(r||(r=(0,o.Z)(["artifact:editor.curSubEff"])))}),(0,W.jsx)(B.Z,{value:qe,max:900,valid:Be})]}),qe!==ze&&(0,W.jsxs)(v.Z,{sx:{display:"flex",mb:1},children:[(0,W.jsx)(Z.Z,{color:"text.secondary",component:"span",variant:"caption",sx:{flexGrow:1},children:pe(i||(i=(0,o.Z)(["artifact:editor.maxSubEff"])))}),(0,W.jsx)(B.Z,{value:ze,max:900,valid:Be})]}),(0,W.jsx)(v.Z,{flexGrow:1}),ie&&(0,W.jsxs)("strong",{children:["Probability: ",(100*(0,L.B)(Ee,ie)).toFixed(2),"%"]}),(0,W.jsxs)(Z.Z,{color:"success.main",children:[null!==(z=null===Ze||void 0===Ze?void 0:Ze.name)&&void 0!==z?z:"Artifact Set"," ",Ue]})]})]}),(0,W.jsxs)(v.Z,{sx:{p:1,display:"flex",gap:1,justifyContent:"space-between",alignItems:"center"},children:[ye&&fe?(0,W.jsx)(I.Z,{sx:{flexGrow:1},size:"small",showDefault:!0,defaultIcon:(0,W.jsx)(h.Z,{}),defaultText:pe("ui:inventory"),value:Te,onChange:function(e){return ve.setArtLocation(U,e)}}):(0,W.jsx)(M.Z,{location:Te}),ye&&(0,W.jsxs)(g.Z,{sx:{height:"100%"},children:[ce&&(0,W.jsx)(j.Z,{title:(0,W.jsx)(Z.Z,{children:pe(a||(a=(0,o.Z)(["artifact:edit"])))}),placement:"top",arrow:!0,children:(0,W.jsx)(k.Z,{color:"info",size:"small",onClick:we,children:(0,W.jsx)(u.G,{icon:s.Xcf,className:"fa-fw"})})}),ue&&(0,W.jsx)(j.Z,{title:(0,W.jsxs)(v.Z,{children:[(0,W.jsx)(Z.Z,{children:pe(l||(l=(0,o.Z)(["artifact:excludeArtifactTip"])))}),(0,W.jsx)(Z.Z,{children:(0,W.jsx)(_.Z,{color:Re?"error":"success",children:pe("artifact:".concat(Re?"excluded":"included"))})})]}),placement:"top",arrow:!0,children:(0,W.jsx)(k.Z,{onClick:function(){return ve.updateArt({exclude:!Re},Ie)},color:Re?"error":"success",size:"small",children:(0,W.jsx)(u.G,{icon:Re?s.gPx:s.Stf,className:"fa-fw"})})}),!!Y&&(0,W.jsx)(k.Z,{color:"error",size:"small",onClick:function(){return Y(Ie)},disabled:Me,children:(0,W.jsx)(u.G,{icon:s.I7k,className:"fa-fw"})}),he]})]})]})]})}function J(e){var t,n,r,i=e.stat,a=e.effFilter,l=e.rarity;if(!i.value)return null;var o=null!==(t=null===(n=i.rolls)||void 0===n?void 0:n.length)&&void 0!==t?t:0,c=i.key?O.Z.maxSubstatValues(i.key):0,s=i.key?O.Z.getSubstatRollData(i.key,l):[],u=7-s.length,d="roll".concat((0,z.uZ)(o,1,6)),f=null!==(r=i.efficiency)&&void 0!==r?r:0,h=(0,z.V2)(.5+f/500*.5),p=V.ZP.getStr(i.key),x=V.ZP.unit(i.key),y=i.key&&a.has(i.key);return(0,W.jsxs)(v.Z,{display:"flex",gap:1,alignContent:"center",children:[(0,W.jsxs)(Z.Z,{sx:{flexGrow:1},color:o?"".concat(d,".main"):"error.main",component:"span",children:[R.Z[i.key]," ",p,"+".concat((0,V.qs)(i.value,V.ZP.unit(i.key))).concat(x)]}),y&&(0,W.jsx)(v.Z,{display:"flex",gap:.25,height:"1.3em",children:i.rolls.sort().map((function(e,t){return(0,W.jsx)(Q,{value:100*e/c,color:"roll".concat((0,z.uZ)(u+s.indexOf(e),1,6),".main")},"".concat(t).concat(e))}))}),(0,W.jsx)(Z.Z,{sx:{opacity:h,minWidth:40,textAlign:"right"},children:y?"".concat(f.toFixed(),"%"):"-"})]})}function Q(e){var t=e.color,n=void 0===t?"red":t,r=e.value,i=void 0===r?50:r;return(0,W.jsx)(v.Z,{sx:{width:7,height:"100%",bgcolor:n,overflow:"hidden",borderRadius:1,display:"inline-block"},children:(0,W.jsx)(v.Z,{sx:{width:10,height:"".concat(100-(0,z.uZ)(i,0,100),"%"),bgcolor:"gray"}})})}},44824:function(e,t,n){n.d(t,{Af:function(){return u},EM:function(){return h},OQ:function(){return c},bq:function(){return s},sZ:function(){return d},x3:function(){return f}});var r=n(37762),i=n(93433),a=n(24351),l=n(19272),o=n(44217),c=["rarity","level","artsetkey","efficiency","mefficiency","probability"],s=["probability"];function u(){return{artSetKeys:[],rarity:(0,i.Z)(a.En),levelLow:0,levelHigh:20,slotKeys:(0,i.Z)(a.eV),mainStatKeys:[],substats:[],location:"",exclusion:["excluded","included"],locked:["locked","unlocked"]}}var d=function(){return{filterOption:u(),ascending:!1,sortType:c[0]}};function f(e,t){return{rarity:{getValue:function(e){var t;return null!==(t=e.rarity)&&void 0!==t?t:0},tieBreaker:"level"},level:{getValue:function(e){var t;return null!==(t=e.level)&&void 0!==t?t:0},tieBreaker:"artsetkey"},artsetkey:{getValue:function(e){var t;return null!==(t=e.setKey)&&void 0!==t?t:""},tieBreaker:"level"},efficiency:{getValue:function(t){return l.Z.getArtifactEfficiency(t,e).currentEfficiency}},mefficiency:{getValue:function(t){return l.Z.getArtifactEfficiency(t,e).maxEfficiency}},probability:{getValue:function(e){if(!Object.keys(t).length)return 0;var n=e.probability;return void 0===n?(0,o.B)(e,t):n}}}}function h(){return{exclusion:function(e,t){return!(!t.includes("included")&&!e.exclude)&&!(!t.includes("excluded")&&e.exclude)},locked:function(e,t){return!(!t.includes("locked")&&e.lock)&&!(!t.includes("unlocked")&&!e.lock)},location:function(e,t){return!t||("Inventory"===t&&!e.location||(!("Equipped"!==t||!e.location)||t===e.location))},artSetKeys:function(e,t){return!t.length||t.includes(e.setKey)},slotKeys:function(e,t){return t.includes(e.slotKey)},mainStatKeys:function(e,t){return!t.length||t.includes(e.mainStatKey)},levelLow:function(e,t){return t<=e.level},levelHigh:function(e,t){return t>=e.level},rarity:function(e,t){return t.includes(e.rarity)},substats:function(e,t){var n,i=(0,r.Z)(t);try{var a=function(){var t=n.value;if(t&&!e.substats.some((function(e){return e.key===t})))return{v:!1}};for(i.s();!(n=i.n()).done;){var l=a();if("object"===typeof l)return l.v}}catch(o){i.e(o)}finally{i.f()}return!0}}}},46797:function(e,t,n){n.d(t,{Z:function(){return o}});var r=n(29439),i=n(25617),a=n(60393),l=n(80184);function o(e){var t=e.value,n=e.max,o=void 0===n?1:n,c=e.valid,s="number"===typeof t?["roll".concat((0,a.uZ)(Math.floor(t/o*10)-4,1,6)),t.toFixed()+"%"]:["secondary",t],u=(0,r.Z)(s,2),d=u[0],f=u[1];return c||(d="error"),(0,l.jsx)(i.Z,{color:d,children:f})}},44217:function(e,t,n){n.d(t,{B:function(){return g}});var r=n(29439),i=n(37762),a=n(4942),l=n(1413),o=n(93433),c=n(60393),s=n(19272),u=n(12354),d=[3,4,6],f={hp:6,atk:6,def:6,hp_:4,atk_:4,def_:4,eleMas:4,enerRech_:4,critRate_:3,critDMG_:3},h={};function p(e,t,n,r){if(5!==e.length)for(var i=0,s=d;i<s.length;i++){var u=s[i];t[u]>0&&p([].concat((0,o.Z)(e),[u]),(0,l.Z)((0,l.Z)({},t),{},(0,a.Z)({},u,t[u]-u)),n-u,r*t[u]/n)}else(0,c.SR)(h,e,r)}p([0],{3:6,4:20,6:18},44,1),p([3],{3:3,4:20,6:18},41,1),p([4],{3:6,4:16,6:18},40,1),p([6],{3:6,4:20,6:12},38,1);for(var v=Array(6).fill(0).map((function(e,t){for(var n=[1],r=0,i=1;++r<=t;)i*=t-r+1,i/=r,n.push(i);return n})),x=[[1]],Z=function(){var e=x[x.length-1],t=Array(e.length+3).fill(0);e.forEach((function(e,n){for(var r=0,i=[0,1,2,3];r<i.length;r++){t[n+i[r]]+=e}})),x.push(t.map((function(e){return e/4})))};x.length<6;)Z();for(var y=function(){var e=b[m],t=e.reduce((function(e,t){return e+t}));e.forEach((function(e,n,r){r[n]=t,t-=e}))},m=0,b=x;m<b.length;m++)y();function g(e,t){if(e.rarity<=2)return NaN;var n=e.rarity,o=e.level,p=e.substats,Z=(0,l.Z)({},t),y=new Set(Object.keys(Z)),m=0,b=e.mainStatKey;if(b in Z){var g=4*n;if(u[n][b][g]<Z[b])return 0;delete Z[b],y.delete(b)}var j,k=(0,i.Z)(p);try{for(k.s();!(j=k.n()).done;){var w=j.value,C=w.key,S=w.value;C?y.has(C)&&(y.delete(C),Z[C]>S?Z[C]-=S:delete Z[C]):m+=1}}catch(T){k.e(T)}finally{k.f()}if(m+=4-p.length,y.size>m||Object.keys(Z).length>4)return 0;for(var E=s.Z.rollsRemaining(o,n)-m,I=0,M=Object.entries(Z);I<M.length;I++){var _=(0,r.Z)(M[I],2),F=_[0],A=_[1];Z[F]=Math.max(Math.ceil(10*A/s.Z.maxSubstatValues(F,n)),1)}var K=0,P=Object.entries(Z).map((function(e){var t=(0,r.Z)(e,2),n=t[0],i=t[1],a=y.has(n)?1:0,l=Math.ceil(i/10)-a;return K+=l,{target:i,filler:a,minUpgrade:l}})).reverse();if(K>E)return 0;var R=(0,a.Z)({},E,1),O=E-K;return P.forEach((function(e,t){for(var n,i,a,l=e.target,o=e.filler,c=e.minUpgrade,s={},u=c;u<=c+O;u++)for(var d=l-7*(u+o),f=d>0?x[u+o][d]:1,h=0,p=Object.entries(R);h<p.length;h++){var Z,y=(0,r.Z)(p[h],2),m=y[0],b=y[1],g=parseInt(m);if(!(g<u)){var j=(i=u,a=4-t,v[n=g][i]*Math.pow(a-1,n-i)/Math.pow(a,n)),k=g-u;s[k]=(null!==(Z=s[k])&&void 0!==Z?Z:0)+b*f*j}}R=s})),function(e,t,n){var r,a,l=null!==(r=f[e])&&void 0!==r?r:0,o=0,s={3:2,4:5,6:3},u=h[l],p=(0,i.Z)(t);try{for(p.s();!(a=p.n()).done;){var x=a.value.key;if(x){var Z=f[x];u=u[Z],s[Z]-=1}}}catch(T){p.e(T)}finally{p.f()}l&&(s[l]-=1);var y,m={3:0,4:0,6:0},b=(0,i.Z)(n);try{for(b.s();!(y=b.n()).done;){var g=y.value;m[f[g]]+=1}}catch(T){b.e(T)}finally{b.f()}var j=0;(0,c.Q1)(u,[],(function(e){return"number"===typeof e}),(function(e,t){j+=e;var n,r={3:0,4:0,6:0},a=(0,i.Z)(t);try{for(a.s();!(n=a.n()).done;){r[n.value]+=1}}catch(T){a.e(T)}finally{a.f()}var l,c=e,s=(0,i.Z)(d);try{for(s.s();!(l=s.n()).done;){var u=l.value,f=r[u],h=m[u];if(f<h)return;c*=v[f][h]}}catch(T){s.e(T)}finally{s.f()}o+=c}));var k,w=(0,i.Z)(d);try{for(w.s();!(k=w.n()).done;){var C=k.value;o/=v[s[C]][m[C]]}}catch(T){w.e(T)}finally{w.f()}return o/j}(e.mainStatKey,p,y)*Object.values(R).reduce((function(e,t){return e+t}))}},63372:function(e,t,n){n.d(t,{Z:function(){return l}});var r=n(29439),i=n(72791),a=n(56928);function l(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:"",t=(0,i.useContext)(a.t),n=t.database,l=(0,i.useState)(n._getArt(e)),o=(0,r.Z)(l,2),c=o[0],s=o[1];return(0,i.useEffect)((function(){return s(n._getArt(e))}),[n,e]),(0,i.useEffect)((function(){return e?n.followArt(e,s):void 0}),[e,s,n]),c}}}]);
//# sourceMappingURL=328.5551de1b.chunk.js.map